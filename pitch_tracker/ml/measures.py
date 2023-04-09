import torch
from mir_eval.melody import to_cent_voicing, evaluate
from pitch_tracker.utils.audio import midi_to_hz

from pitch_tracker.utils.constants import MIDI_START, MIDI_END, F_MIN

def class_to_frequency(class_inputs:torch.Tensor, midi_start=MIDI_START, n_classes=89):
    # n_classes also includes non-melody pitches
    pre_midi_start = midi_start - 1
    voiced_mask = class_inputs != 0

    midi_values = class_inputs + pre_midi_start
    output_frequencies = midi_to_hz(midi_values)
    output_frequencies = output_frequencies * voiced_mask

    return output_frequencies

def melody_evaluate(y_true:torch.Tensor, y_pred:torch.Tensor):
    # one-hot y size: [n_batches, n_frames, n_classes]
    # argmax y size: [n_batches, n_frames]
    y_true, y_pred = y_true.cpu(), y_pred.cpu()

    if len(y_true.shape) == 3:
        y_true = torch.argmax(y_true, dim=-1)
        y_pred = torch.argmax(y_pred, dim=-1)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    ref_freq = class_to_frequency(y_true)
    est_freq = class_to_frequency(y_pred)
    time_1d = torch.arange(0, ref_freq.numel(),1)

    scores = evaluate(
        ref_time=time_1d.numpy(),
        ref_freq=ref_freq.numpy(),
        est_time=time_1d.numpy(),
        est_freq=est_freq.numpy(),
        cent_tolerance=51
        )
    
    return scores 
