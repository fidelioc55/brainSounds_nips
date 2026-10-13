"""
AudioNeuroDataset
=================
Input data format:
  pooled_data[roi][cv]["X_train"]      → (N, 1024)  float32  fMRI beta values per ROI
  pooled_data[roi][cv]["train_sounds"] → (N,)       int      1-based sound index

After stacking over roi_list:
  roi_pooled_train  → (N, num_rois, 1024)   ← this is [B, P, V]
  train_wav_paths   → list of N .wav paths

__getitem__ returns:
  {
    "brain_data":    Tensor [num_rois, max_voxels]   (already padded/truncated)
    "audio_target":  Tensor [samples]                (normalized waveform)
    "sound_idx":     int
  }
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import re
from typing import Optional


ROI_LIST = ['HG2hem', 'PT2hem', 'PP2hem', 'mSTG2hem', 'pSTG2hem', 'aSTG2hem']


class BrainNormalizer:
    """Dataset-level z-score normalizer fit on the training split only."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, roi_data: np.ndarray, eps: float = 1e-6):
        mean = torch.from_numpy(roi_data.mean(axis=0).astype(np.float32))
        std = torch.from_numpy(roi_data.std(axis=0).astype(np.float32)).clamp_min(eps)
        return cls(mean=mean, std=std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def soundname_to_wav(sound_name: str) -> str:
    """
    Converts a string such as:
    'stim145_cat04_music_exemp01'
    into:
    's2_music_1.wav'
    """
    m = re.match(r"stim\d+_cat\d+_([a-zA-Z]+)_exemp(\d+)", sound_name)
    if m is None:
        raise ValueError(f"Unexpected format: {sound_name}")

    category = m.group(1).lower()
    exemp = int(m.group(2))

    return f"s2_{category}_{exemp}.wav"


def average_test_by_sound(X_test, test_sounds):
    """
    Average X_test across subjects for each test sound.

    Input:
        X_test: (n_subjects * 72, n_voxels)
        test_sounds: (n_subjects * 72,)

    Output:
        X_test_avg: (72, n_voxels)
        test_sounds_avg: (72,)
    """
    unique_sounds = np.unique(test_sounds)

    X_test_avg = []
    test_sounds_avg = []

    for s in unique_sounds:
        idx = np.where(test_sounds == s)[0]
        X_test_avg.append(X_test[idx].mean(axis=0))
        test_sounds_avg.append(s)

    X_test_avg = np.vstack(X_test_avg)
    test_sounds_avg = np.array(test_sounds_avg)

    return X_test_avg, test_sounds_avg


def average_roi_data_by_sound(roi_data: np.ndarray, sound_idxs: np.ndarray) -> tuple:
    """
    Average repeated test measurements for each sound across the first axis.

    Args:
        roi_data: np.ndarray [N, num_rois, num_voxels]
        sound_idxs: np.ndarray [N,]

    Returns:
        roi_data_avg: np.ndarray [num_unique_sounds, num_rois, num_voxels]
        sound_idxs_avg: np.ndarray [num_unique_sounds,]
    """
    unique_sounds = np.unique(sound_idxs)
    roi_data_avg = []

    for sound_idx in unique_sounds:
        idx = np.where(sound_idxs == sound_idx)[0]
        roi_data_avg.append(roi_data[idx].mean(axis=0))

    roi_data_avg = np.stack(roi_data_avg, axis=0).astype(np.float32)
    return roi_data_avg, unique_sounds.astype(sound_idxs.dtype)


def load_pooled_data(
    pooled_data: dict,
    cv: str,
    roi_list: list = ROI_LIST,
    split: str = "train",
) -> tuple:
    """
    Loads and stacks data for all ROIs.

    Returns:
        roi_data   : np.ndarray  [N, num_rois, 1024]
        sound_idxs : np.ndarray  [N,]  (0-based indices)
    """
    assert split in ("train", "test")
    key_x     = f"X_{split}"
    key_sounds = f"{split}_sounds"

    stacked = []
    sound_idxs = None

    for reg in roi_list:
        entry = pooled_data[reg][cv]
        stacked.append(entry[key_x])                         # (N, 1024)
        if sound_idxs is None:
            sound_idxs = entry[key_sounds].astype(int) - 1   # → 0-based

    roi_data = np.stack(stacked, axis=1)                     # (N, num_rois, 1024)
    return roi_data, sound_idxs


class AudioNeuroDataset(Dataset):
    """
    Dataset for brain-to-audio decoding.

    Args:
        roi_data      : np.ndarray [N, num_rois, 1024]
        sound_idxs    : np.ndarray [N,]  0-based sound indices
        wav_paths     : list[str]  paths to audio files indexed by sound_idxs
        target_sr     : int        target sample rate (Stable Audio: 44100)
        target_len_s  : float      duration in seconds to extract / pad
        normalize_brain : bool     per-sample z-score normalization of neural data
    """

    def __init__(
        self,
        roi_data: np.ndarray,
        sound_idxs: np.ndarray,
        wav_paths: list,
        target_sr: int = 44100,
        target_len_s: float = 4.0,
        brain_normalizer: Optional[BrainNormalizer] = None,
    ):
        assert roi_data.shape[0] == len(sound_idxs), \
            f"Mismatch: roi_data {roi_data.shape[0]} vs sound_idxs {len(sound_idxs)}"

        self.roi_data       = torch.from_numpy(roi_data.astype(np.float32))  # [N, P, 1024]
        self.sound_idxs     = sound_idxs
        self.wav_paths      = wav_paths
        self.target_sr      = target_sr
        self.target_samples = int(target_sr * target_len_s)
        self.brain_normalizer = brain_normalizer

        self.num_rois    = roi_data.shape[1]   # 6
        self.max_voxels  = roi_data.shape[2]   # 1024
        self.brain_normalizer = brain_normalizer

    def _load_audio(self, path: str) -> torch.Tensor:
        """
        Loads a .wav file, converts it to mono, resamples it to target_sr,
        and adjusts it to exactly target_samples samples.

        Returns: Tensor [target_samples]
        """
        wav, sr = torchaudio.load(path)

        # Stereo → mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        wav = wav.squeeze(0)   # [samples]

        # Pad or truncate to the target length
        if wav.shape[0] < self.target_samples:
            pad = self.target_samples - wav.shape[0]
            wav = F.pad(wav, (0, pad))
        else:
            wav = wav[:self.target_samples]

        # Normalize to [-1, 1]
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak

        return wav   # [target_samples]

    def __len__(self) -> int:
        return len(self.sound_idxs)

    def __getitem__(self, idx: int) -> dict:
        brain = self.roi_data[idx]           # [num_rois, 1024]

        if self.brain_normalizer is not None:
            brain = self.brain_normalizer.normalize(brain)

        sound_idx = self.sound_idxs[idx]
        audio     = self._load_audio(self.wav_paths[sound_idx])   # [target_samples]

        return {
            "brain_data":  brain,       # [6, 1024]
            "audio_target": audio,      # [target_samples]
            "sound_idx":   int(sound_idx),
        }


def build_datasets(
    pooled_data: dict,
    cv: str,
    wav_dir: str,
    sound_names: np.ndarray,
    roi_list: list = ROI_LIST,
    target_sr: int = 44100,
    target_len_s: float = 4.0,
    average_test_repeats: bool = False,
    brain_normalizer_override: Optional[BrainNormalizer] = None,
) -> tuple:
    """
    Factory function: creates train and test datasets in one step.

    Args:
        pooled_data  : raw neural data dictionary
        cv           : cross-validation key, e.g. "CV2"
        wav_dir      : root folder of the .wav files
        sound_names  : np.ndarray of sound names (SoundNames.npy)
        brain_normalizer_override : if provided, e.g. loaded from a checkpoint,
                                    it is used instead of recomputing it from train data.

    Returns:
        train_dataset, test_dataset
    """
    all_wav_paths = [
        os.path.join(wav_dir, soundname_to_wav(n))
        for n in sound_names
    ]

    train_roi, train_idx = load_pooled_data(pooled_data, cv, roi_list, split="train")
    test_roi,  test_idx  = load_pooled_data(pooled_data, cv, roi_list, split="test")
    if average_test_repeats:
        test_roi, test_idx = average_roi_data_by_sound(test_roi, test_idx)

    if brain_normalizer_override is not None:
        brain_normalizer = brain_normalizer_override
    else:
        brain_normalizer = BrainNormalizer.fit(train_roi)

    train_ds = AudioNeuroDataset(
        train_roi,
        train_idx,
        all_wav_paths,
        target_sr,
        target_len_s,
        brain_normalizer=brain_normalizer,
    )
    test_ds = AudioNeuroDataset(
        test_roi,
        test_idx,
        all_wav_paths,
        target_sr,
        target_len_s,
        brain_normalizer=brain_normalizer,
    )

    print(f"Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")
    print(f"  brain_data shape: {train_ds.num_rois} ROIs × {train_ds.max_voxels} voxels")
    print(f"  audio_target:     {train_ds.target_samples} samples @ {target_sr}Hz "
          f"({target_len_s}s)")

    return train_ds, test_ds