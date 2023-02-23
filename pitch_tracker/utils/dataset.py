import os
import csv
from typing import Any, Dict, Generator, List, Tuple, Union

import librosa
import numpy as np
import torch
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram

from medleydb import load_all_multitracks
# from pitch_tracker.utils import files
from pitch_tracker.utils.audio import load_audio_mono
from pitch_tracker.utils.constants import (ONSET_TIME_THRESHOLD,
                                           PICKING_FRAME_SIZE, PRE_MIDI_START)

from pitch_tracker.utils.files import list_file_paths_in_dir, get_file_name, save_pickle

DIST_THRESHOLD = 0.1
EMPTY_THRESHOLD = PICKING_FRAME_SIZE / 5


def create_label_dict_from_dir(label_dir: str) -> Dict[str, np.ndarray]:
    """Creates a dictionary that maps label names to their corresponding note messages in numpy arrays.

    Args:
        label_folder (str): The directory where the label files are located.

    Returns:
        dict: A dictionary that maps label names to their note messages as numpy arrays.
    """
    label_dict = {}
    file_paths = list_file_paths_in_dir(label_dir)
    for label_path in file_paths:
        note_messages = read_label_file(label_path)
        label_name = get_file_name(label_path, include_ext=False)
        label_dict[label_name] = note_messages
    return label_dict


# to be deprecated
def create_label_dict_from_path_list(path_list: list):
    label_dict = {}
    for label_path in path_list:
        note_messages = read_label_file(label_path)
        label_name = get_file_name(label_path, include_ext=False)
        label_dict[label_name] = note_messages
    return label_dict


def create_audio_path_dict() -> Dict[str, str]:
    """Return the MedleyDB Audio path as a dictionary, song_name: audio_mix_path

    Returns:
        Dict[str, str]: {song_name: audio_mix_path}
    """
    audio_path_dict = {}
    mtracks = load_all_multitracks()
    for mt in mtracks:
        audio_path_dict[mt.track_id] = mt.mix_path
    return audio_path_dict


def create_label_path_dict(label_dir: str) -> Dict[str, str]:
    """Return a dictionary contain the song_name: label_path

    Args:
        label_folder (str): the path to the label directory.

    Returns:
        Dict[str,str]: {song_name: label_path}
    """
    label_path_dict = {}
    file_paths = list_file_paths_in_dir(label_dir)
    for file_path in file_paths:
        label_name = get_file_name(file_path, include_ext=False)
        label_path_dict[label_name] = file_path
    return label_path_dict


def create_dataset_path_dict(label_dir: str) -> Dict[str, Tuple[str, str]]:
    """Return a dictionary contain the {song_name: (label_path, audio_mix_path)}

    Args:
        label_folder (str): the path to the label directory.

    Returns:
        Dict[str,str]: {song_name: label_path, audio_mix_path}
    """
    dataset_paths = {}
    audio_path_dict = create_audio_path_dict()
    label_path_dict = create_label_path_dict(label_dir)

    for label_name in label_path_dict.keys():
        audio_path = audio_path_dict[label_name]
        label_path = label_path_dict[label_name]
        dataset_paths[label_name] = (label_path, audio_path)
    return dataset_paths


def build_pick_features_and_time(
        STFT_features: np.ndarray,
        picking_frame_step: int,
        picking_frame_size: int,
        step_frame: int,
        step_time: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

    n_frames = STFT_features.shape[0]
    n_win_frames = int(np.ceil(n_frames / step_frame / picking_frame_step))

    times_1D = np.arange(0, n_win_frames * picking_frame_size) * step_time
    pick_features = np.zeros(
        (n_win_frames, picking_frame_size * step_frame, STFT_features.shape[1]))
    pick_times = np.zeros((n_win_frames, picking_frame_size))

    for i, j in enumerate(range(0, n_frames, picking_frame_step*step_frame)):
        pick_times[i, :] = times_1D[int(
            j/step_frame):int(j/step_frame)+picking_frame_size]
        end_idx = j + picking_frame_size * step_frame
        end_idx = end_idx if end_idx < n_frames else n_frames
        copy_size = end_idx - j
        pick_features[i, :copy_size, :] = STFT_features[range(j, end_idx), :]

    return [pick_features, pick_times]


def create_label_generator(
        note_messages: np.ndarray,
        feature_times: np.ndarray,
        dist_threshold: float,
        empty_threshold: float,
        onset_time_threshold: float,
        picking_frame_size: int,
        step_time: int,
        n_class: int,
        pre_midi_start: int = PRE_MIDI_START) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:

    for t_frame in feature_times:
        stime = t_frame[0] - step_time
        etime = t_frame[-1] + step_time

        note_messages_in_frame = _get_note_messages_in_frame(
            note_messages, stime, etime)

        onset_labels = _get_onset_label(
            note_messages_in_frame,
            t_frame,
            picking_frame_size,
            onset_time_threshold
        )

        duration_labels = _get_duration_label(
            t_frame,
            note_messages_in_frame,
            onset_labels,
            dist_threshold,
            empty_threshold,
            picking_frame_size
        )

        pitch_labels = _get_pitch_label(
            t_frame,
            note_messages_in_frame,
            picking_frame_size=picking_frame_size,
            n_class=n_class,
            pre_midi_start=pre_midi_start
        )

        yield [onset_labels, duration_labels, pitch_labels]


def extract_stft_feature(
        y: np.ndarray,
        n_fft: int,
        hop_length: int,
        mean: float = 0.0,
        var: float = 1.0) -> np.ndarray:

    stft_feature = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    log_compress = librosa.core.power_to_db(np.square(np.abs(stft_feature)))
    return (log_compress - mean) / var


def extract_melspectrogram_feature(
        y: np.ndarray,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        sample_rate: int = 44100,
        win_length: int = None,
        fmin: float = 0.0,
        mean: float = 0.0,
        var: float = 1.0,
        backend: str = 'torch') -> Union[np.ndarray, torch.Tensor]:
    """extract the melspectrogram feature from the audio signal.

    Args:
        y (np.ndarray): the audio signal, shape: [n_channels, n_samples] or [n_samples]
        n_fft (int): the length of the FFT window.
        hop_length (int): the number of samples between successive frame.
        n_mels (int): the number of mel bands.
        sample_rate (int, optional): the sample rate of the audio signal. Defaults to 44100.
        win_length (int, optional): the length of the windows function, not to be confused with n_fft. Defaults to None.
        fmin (float, optional): the lowest bound of the frequency. Defaults to 0.0.
        mean (float, optional): the mean of the dataset, used for normalizing. Defaults to 0.0.
        var (float, optional): the var of the dataset, used for normalizing. Defaults to 1.0.
        backend (str, optional): either `librosa` or `torchaudio`. Defaults to 'librosa'.

    Returns:
        np.ndarray | torch.Tensor: the melspectrogram feature, shape: [n_channels, n_mels, n_frames] 
    """
    if backend == 'torch':
        melspectrogram_extractor = MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            pad_mode='constant',
            center=True,
            f_min=librosa.midi_to_hz(21),
            f_max=None,
            n_mels=n_mels,
            power=2.0,
            norm="slaney",
        )
        melspectrogram_feature = melspectrogram_extractor(y)
        log_compress = amplitude_to_DB(
            melspectrogram_feature, multiplier=10., amin=1e-10, db_multiplier=1.0)

    else:
        melspectrogram_feature = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            pad_mode='constant',
            center=True,
            fmin=fmin,
            fmax=None,
            n_mels=n_mels,
            power=2.0,
            htk=True
        )
        log_compress = librosa.power_to_db(melspectrogram_feature)
    return (log_compress - mean) / var


def read_label_file(label_path: str):
    with open(label_path, 'r') as f:
        csv_reader = csv.reader(f)
        note_messages = []
        for line in csv_reader:
            stime = float(line[0])
            etime = float(line[1])
            pitch = float(line[2])
            note_messages.append((stime, etime, pitch))
    note_messages = np.asarray(note_messages, dtype=np.float32)
    return note_messages


def create_feature_label_generator(
        dataset_path_dict: Dict[str, Tuple[str, str]],
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        n_class: int,
        hop_length: int,
        picking_frame_step: int,
        picking_frame_size: int,
        step_frame: int,
        step_time: int,
        dist_threshold: float,
        empty_threshold: float):

    # label_paths = (label_path for label_path, _ in dataset_path_dict.values())
    # audio_paths = (audio_path for _, audio_path in dataset_path_dict.values())
    # label_dict = create_label_dict_from_path_list(label_paths)

    for label_name in dataset_path_dict:
        label_path, audio_path = dataset_path_dict[label_name]
        if not audio_path:
            print(f'No audio: {label_name}')
            continue
        note_messages = read_label_file(label_path)

        end_time_label = note_messages[-1, 1] - 0.1
        # logger.info(f'{label_name} end: {end_time_label}')

        try:
            signal, sr = load_audio_mono(
                audio_path, sample_rate, keep_channel_dim=False)
        except Exception as e:
            print(f'Falied to load {label_name} - {e}')
            continue

        eindex = int(end_time_label * sr)

        melspectrogram_features = extract_melspectrogram_feature(
            signal[:eindex],
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            backend='librosa'
        )

        pick_features, pick_times = build_pick_features_and_time(
            melspectrogram_features.T,
            picking_frame_step,
            picking_frame_size,
            step_frame,
            step_time
        )

        label_generator = create_label_generator(
            note_messages=note_messages,
            feature_times=pick_times,
            dist_threshold=dist_threshold,
            empty_threshold=empty_threshold,
            onset_time_threshold=ONSET_TIME_THRESHOLD,
            picking_frame_size=picking_frame_size,
            step_time=step_time,
            n_class=n_class,
            pre_midi_start=PRE_MIDI_START,
        )

        feature_generator = (feature for feature in pick_features)

        # feature_and_label = _intergen(feature_generator, label_generator)
        feature_and_label = ((feature, label) for feature,
                             label in zip(feature_generator, label_generator))

        yield label_name, feature_and_label


def write_feature_label_to_disk(feature_label_generator: Generator, output_dir: str, is_overwrite: bool = True) -> List[str]:
    """
    Writes feature and label pairs to disk as pickle files by song. Output <song_name>.pkl.

    Args:
        feature_label_generator (Generator): A generator that yields tuples of (label, feature_label_pairs).
        output_dir (str): The directory to save the pickle files.
        is_overwrite (bool, optional): If True, overwrite the file if it already exists. Defaults to True.

    Returns:
        List[str]: A list of the passed files.

    Raises:
        IOError: If the output directory does not exist.

    Example:
        feature_label_generator = get_feature_label_generator()
        output_dir = './data'
        write_feature_label_to_disk(feature_label_generator, output_dir)
    """
    passed_files = []
    for label_name, feature_label_pair in feature_label_generator:
        output_path = os.path.join(output_dir, label_name + '.pkl')

        if os.path.exists(output_path) and not is_overwrite:
            print(f'{output_path} already existed, skipping.')
            continue

        feature_batch = []
        label_batch = {}
        onsets = []
        durations = []
        pitches = []
        for feature, (onset, duration, pitch) in feature_label_pair:
            feature_batch.append(feature)
            onsets.append(onset)
            durations.append(duration)
            pitches.append(pitch)
        feature_batch = np.array(feature_batch)
        label_batch['onset'] = np.array(onsets)
        label_batch['duration'] = np.array(durations)
        label_batch['pitch'] = np.array(pitches)
        save_pickle([feature_batch, label_batch],
                    output_path, is_overwrite=is_overwrite)
        passed_files.append(label_name)
    return passed_files

def write_feature_label_to_disk_by_frame(
    feature_label_generator: Generator,
    output_dir: str,
    is_overwrite: bool = True,
    categorize_by_subdir:bool=False) -> List[str]:
    """
    Writes feature and label pairs to disk as pickle files by frame.
    Output <song_name>.pkl.

    Args:
        feature_label_generator (Generator): A generator that yields tuples of (label, feature_label_pairs).
        output_dir (str): The directory to save the pickle files.
        is_overwrite (bool, optional): If True, overwrite the file if it already exists. Defaults to True.
        categorize_by_subdir (bool, optional): If True, create a sub-directory to contain the feature & label for each song. Defaults to False.
    Returns:
        List[str]: A list of the passed files.

    Raises:
        IOError: If the output directory does not exist.

    Example:
        feature_label_generator = get_feature_label_generator()
        output_dir = './data'
        write_feature_label_to_disk(feature_label_generator, output_dir)
    """

    # TODO: fix output path so the function save each frame as a pickle file.
    passed_files = []
    for label_name, feature_label_pair in feature_label_generator:
        
        song_dir = output_dir
        if categorize_by_subdir:
            song_dir = os.path.join(output_dir, label_name)
            os.makedirs(song_dir, exist_ok=True)
        

        # feature_batch = []
        # label_batch = {}
        # onsets = []
        # durations = []
        # pitches = []
        for i, (feature, label) in enumerate(feature_label_pair):
            output_path = os.path.join(song_dir, f'{label_name}_{i}.pkl')
            if os.path.exists(output_path) and not is_overwrite:
                print(f'{output_path} already existed, skipping.')
                continue
            save_pickle(
                [feature, label],
                output_path,
                is_overwrite=is_overwrite)

        #     feature_batch.append(feature)
        #     onsets.append(onset)
        #     durations.append(duration)
        #     pitches.append(pitch)
        # feature_batch = np.array(feature_batch)
        # label_batch['onset'] = np.array(onsets)
        # label_batch['duration'] = np.array(durations)
        # label_batch['pitch'] = np.array(pitches)
        # save_pickle([feature_batch, label_batch],
        #             output_path, is_overwrite=is_overwrite)
        passed_files.append(label_name)
    return passed_files


def _get_note_messages_in_frame(note_messages: np.ndarray, stime, etime):
    stimes = note_messages[:, 0]
    note_messages_in_frame_idices = np.flatnonzero(
        (stimes >= stime) & (stimes <= etime))
    notes_in_frame = note_messages[note_messages_in_frame_idices, :]
    return notes_in_frame


def _get_onset_label(
        note_messages_in_frame: np.ndarray,
        t_frame,
        picking_frame_size: int,
        onset_time_threshold: float):

    onset_labels = np.zeros(picking_frame_size)
    for of in note_messages_in_frame[:, 0]:
        diff = np.abs(of - t_frame)
        onset_idx = diff < onset_time_threshold
        onset_labels[onset_idx] = 1

    return onset_labels


def _get_duration_label(
        t_frame,
        note_messages_in_frame: np.ndarray,
        onset_labels,
        dist_threshold: float,
        empty_threshold: float,
        picking_frame_size: int):
    duration_labels = np.zeros(picking_frame_size)
    for i, t in enumerate(t_frame):
        if onset_labels[i]:
            duration_labels[i] = 1
            continue
        stime = t + dist_threshold
        etime = t - dist_threshold
        is_in_notes = np.flatnonzero(
            (note_messages_in_frame[:, 0] <= stime) & (note_messages_in_frame[:, 1] >= etime))
        if is_in_notes.size > 0:
            duration_labels[i] = 1

        num_non_silence = np.flatnonzero(duration_labels).size
        if num_non_silence < empty_threshold:
            # Has too many silence frames so we keep (skip?) this super frame
            continue

    return duration_labels


def _get_pitch_label(
        t_frame,
        note_messages_in_frame: np.ndarray,
        picking_frame_size: int,
        n_class: int,
        pre_midi_start: int):
    pitch_labels = np.zeros((picking_frame_size, n_class))
    for i, t in enumerate(t_frame):
        idx = np.flatnonzero(
            (note_messages_in_frame[:, 0] <= t) & (note_messages_in_frame[:, 1] > t))
        if (idx.size > 0):
            last_label_idx = idx[-1]
            note_pitch = int(
                note_messages_in_frame[last_label_idx, 2] - pre_midi_start)
            pitch_labels[i, note_pitch] = 1
        else:
            pitch_labels[i, 0] = 1
    return pitch_labels
