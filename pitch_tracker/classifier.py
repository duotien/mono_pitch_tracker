from functools import partial
from typing import Any, List, Union
import os

import mido
import numpy as np
import torch

from pitch_tracker import THESIS_2023_MODEL_3_PATH
from pitch_tracker.ml.net import MPT2023_3
from pitch_tracker.utils.audio import load_audio_mono
from pitch_tracker.utils.constants import (ANALYSIS_FRAME_SIZE,
                                           ANALYSIS_FRAME_TIME, F_MIN,
                                           HOP_LENGTH, N_FFT, N_MELS,
                                           PATCH_SIZE, PRE_MIDI_START,
                                           SAMPLE_RATE)
from pitch_tracker.utils.dataset import (build_pick_features_and_time,
                                         extract_melspectrogram_feature)
from pitch_tracker.utils.midi import build_note_messages, convert_to_midi


class MelodyExtractor():
    def __init__(
            self,
            model_path=THESIS_2023_MODEL_3_PATH,
            n_fft: int = N_FFT,
            n_mels: int = N_MELS*2,
            hop_length: int = HOP_LENGTH,
            patch_size: int = PATCH_SIZE,
            analysis_frame_size: int = ANALYSIS_FRAME_SIZE,
            analysis_frame_time: float = ANALYSIS_FRAME_TIME,
            fmin: float = F_MIN,
            device='cpu') -> None:

        self.model_path = model_path
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.patch_size = patch_size
        self.analysis_frame_size = analysis_frame_size
        self.analysis_frame_time = analysis_frame_time
        self.fmin = fmin
        self.device = device

        self.model = MPT2023_3().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))

    def __call__(
            self,
            file_path: str = None,
            signal: Union[torch.Tensor, np.ndarray] = None,
            sample_rate: int = SAMPLE_RATE,
            voicing_bias: float = 0.0,
    ) -> Any:

        pick_features, pick_times = self.get_pick_features_and_time(
            file_path = file_path,
            signal = signal,
            sample_rate = sample_rate
        )

        # get model prediction
        self.model.eval()
        pred = self.model(pick_features)
        pred[:, :, 0] -= voicing_bias
        pitch = pred.argmax(2).flatten()
        pitch[pitch > 0] += PRE_MIDI_START

        pick_times = pick_times.flatten()

        return pitch, pick_times

    def to_midi(self, audio_path:str, ticks_per_beat:int=960, voicing_bias:float=0.0):
        pitch, time1d = self(audio_path, voicing_bias=voicing_bias)
        note_sequences = self.build_note_sequences(pitch, ANALYSIS_FRAME_TIME)
        note_messages = build_note_messages(
            note_sequences, ticks_per_beat=ticks_per_beat)
        midi = convert_to_midi(note_messages.numpy(),
                               ticks_per_beat=ticks_per_beat)
        return midi

    def export_to_midi(self, audio_path:str, out_midi_path:str, tick_per_beat:int=960, voicing_bias:float=0.0):
        midi = self.to_midi(audio_path, tick_per_beat, voicing_bias)
        midi.save(out_midi_path)

    def export_to_midis(self, src: Union[str, List[str]], dst_dir: str, tick_per_beat: int = 960, voicing_bias: float = 0.0):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        if isinstance(src, str):
            src = [os.path.join(src, f) for f in os.listdir(src) if f.endswith(('.wav', '.mp3', '.ogg'))]

        for audio_path in src:
            out_midi_path = os.path.join(dst_dir, os.path.splitext(os.path.basename(audio_path))[0] + '.mid')
            try:
                self.export_to_midi(audio_path, out_midi_path, tick_per_beat, voicing_bias)
            except Exception as e:
                print(f'Cannot convert {audio_path} - {e}')

    def get_pick_features_and_time(
            self,
            file_path: str = None,
            signal: Union[torch.Tensor, np.ndarray] = None,
            sample_rate: int = SAMPLE_RATE):

        if file_path is None and signal is None:
            raise Exception(
                'Missing one required parameter `file_path` or `signal`.')

        # ignore `signal` param if file_path is used
        if file_path:
            signal, _ = load_audio_mono(
                file_path, sample_rate, keep_channel_dim=False)

        melspectrogram_features = extract_melspectrogram_feature(
            y = signal,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            n_mels = self.n_mels,
            sample_rate = sample_rate,
            fmin = self.fmin,
            log_compress=True,
            backend = 'librosa',
        )

        # patch_step = patch_size so we don't make overlapping frame
        pick_features, pick_times = build_pick_features_and_time(
            STFT_features = melspectrogram_features.T,
            patch_step = self.patch_size,
            patch_size = self.patch_size,
            analysis_frame_size = self.analysis_frame_size,
            analysis_frame_time = self.analysis_frame_time
        )

        pick_features = torch.from_numpy(pick_features).type(torch.float32)
        pick_times = torch.from_numpy(pick_times).type(torch.float32)

        pick_features = pick_features.unsqueeze(1)
        return pick_features, pick_times

    def get_consecutive_pred(self, pitch_pred: torch.Tensor):
        split_indices = torch.where(torch.diff(pitch_pred) != 0)[0]+1
        split_indices = split_indices.tolist()
        pitch_pred_indices_mask = torch.arange(pitch_pred.shape[0])

        sequences = torch.hsplit(pitch_pred_indices_mask, split_indices)
        sequences_pitch_values = pitch_pred[[
            indices[0] for indices in sequences]].tolist()
        sequences_pitch_values = tuple(sequences_pitch_values)

        return list(zip(sequences, sequences_pitch_values))
    
    def get_first_last_indices_pairs_of_consecutive_pitch_pred(self, pitch_pred: torch.Tensor):
        split_indices = torch.where(torch.diff(pitch_pred) != 0)[0]+1
        split_indices = split_indices.tolist()
        pitch_pred_indices_mask = torch.arange(pitch_pred.size(dim=0))

        sequences = torch.hsplit(pitch_pred_indices_mask, split_indices)

        first_last_indices_pairs = []
        for sequence in sequences:
            first_idx = sequence[0].item()
            last_idx = sequence[-1].item()
            first_last_indices_pairs.append((first_idx, last_idx))

        sequences_pitch_values = pitch_pred[[first_idx for first_idx, _ in first_last_indices_pairs]].tolist()
        sequences_pitch_values = tuple(sequences_pitch_values)

        return list(zip(first_last_indices_pairs, sequences_pitch_values))


    def build_note_sequences(self, pitch_pred: torch.Tensor, analysis_frame_time: int, analysis_frame_powers=None):
        note_sequences = []
        pitch_sequences = self.get_first_last_indices_pairs_of_consecutive_pitch_pred(pitch_pred)
        
        #TODO: Add a way to get power value for each frame
        if analysis_frame_powers is None:
            analysis_frame_powers = 50
        
        # filter non-melody sequences
        pitch_sequences = [(first_last_pairs, midi_value) for first_last_pairs,
                           midi_value in pitch_sequences if midi_value != 0]
        for (first_idx, last_idx), midi_value in pitch_sequences:
            start_time = first_idx * analysis_frame_time
            end_time = (last_idx + 1) * analysis_frame_time
            note_sequences.append(
                (start_time, end_time, midi_value, analysis_frame_powers))

        return torch.Tensor(note_sequences)