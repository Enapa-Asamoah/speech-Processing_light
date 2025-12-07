"""
Data augmentation utilities
"""

import numpy as np
import librosa
import random
import inspect

def augment_audio(y, sr):
    """Apply random audio augmentations: time stretch, pitch shift, and noise addition."""
    # Time Stretch
    if random.random() < 0.5:
        rate = random.uniform(0.8, 1.2)
        # use safe wrapper to avoid signature mismatches in some environments
        try:
            y = librosa.effects.time_stretch(y, rate)
        except TypeError:
            # fallback: use phase_vocoder on STFT
            try:
                D = librosa.stft(y)
                D_stretch = librosa.phase_vocoder(D, rate)
                y = librosa.istft(D_stretch)
            except Exception:
                # give up time-stretch for this sample
                pass

    # Pitch Shift
    if random.random() < 0.5:
        n_steps = random.randint(-2, 2)
        # use safe wrapper to avoid signature mismatches
        try:
            y = librosa.effects.pitch_shift(y, sr, n_steps)
        except TypeError:
            # fallback: approximate pitch shift by resampling
            try:
                rate = 2.0 ** (n_steps / 12.0)
                y_shift = librosa.resample(y, orig_sr=sr, target_sr=int(sr * rate))
                # trim or pad to original length
                if len(y_shift) > len(y):
                    y = y_shift[: len(y)]
                else:
                    y = np.pad(y_shift, (0, max(0, len(y) - len(y_shift))))
            except Exception:
                # give up pitch-shift for this sample
                pass

    # Add Noise
    if random.random() < 0.5:
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])

    return y