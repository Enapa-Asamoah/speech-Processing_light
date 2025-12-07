"""
Audio preprocessing utilities (RAVDESS-stable version)
"""

import os
import numpy as np
try:
    import soundfile as sf
except Exception:
    sf = None
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from .augment import augment_audio
from PIL import Image


def _normalize_and_resize(feature, size):
    """Normalize a 2D feature to 0-255 uint8 and resize to `size` (width, height).

    Args:
        feature: 2D numpy array
        size: (width, height) tuple
    Returns:
        2D uint8 numpy array of shape (height, width)
    """
    # ensure finite
    feature = np.nan_to_num(feature)
    # shift to zero
    fmin = feature.min()
    feature = feature - fmin
    fmax = feature.max()
    if fmax > 0:
        feature = feature / fmax
    else:
        feature = feature * 0.0

    arr = (feature * 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.resize(size, resample=Image.BILINEAR)
    return np.array(img)


class AudioPreprocessor:
    def __init__(
        self,
        sr=16000,
        sample_rate=None,
        duration=3.0,
        segment_length=None,
        n_mels=128,
        n_mfcc=40,
        save_png=True,
        output_dir="data/features",
        augment=False,
    ):
        if sample_rate is not None:
            sr = sample_rate
        if segment_length is not None:
            duration = segment_length

        self.sr = sr
        self.duration = duration
        self.target_len = int(self.sr * self.duration)
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.save_png = save_png
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.augment = augment

    # -----------------------------------------------------
    # RELIABLE AUDIO LOADING FOR RAVDESS
    # -----------------------------------------------------
    def load_audio(self, path):
        """
        Try loading WAV using soundfile first (stable for RAVDESS),
        fallback to librosa if needed.
        """

        # 1. Try using soundfile
        if sf is not None:
            try:
                y, sr = sf.read(path)
                # convert to mono
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)

                # resample if needed
                if sr != self.sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
            except Exception:
                y = None
        else:
            y = None

        # 2. Fallback: try librosa if soundfile wasn't available or failed
        if y is None:
            try:
                y, sr = librosa.load(path, sr=self.sr, mono=True)
            except Exception as e:
                print(f"[WARN] Could not load audio '{path}': {e}")
                return None

        # Trim or pad to fixed duration
        if len(y) < self.target_len:
            y = np.pad(y, (0, self.target_len - len(y)))
        else:
            y = y[:self.target_len]

        return y

    # -----------------------------------------------------
    # MEL-SPECTROGRAM
    # -----------------------------------------------------
    def extract_mel(self, y):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
        )
        return librosa.power_to_db(mel, ref=np.max)

    # -----------------------------------------------------
    # MFCC
    # -----------------------------------------------------
    def extract_mfcc(self, y):
        return librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=self.n_mfcc
        )

    # -----------------------------------------------------
    # CHROMA
    # -----------------------------------------------------
    def extract_chroma(self, y):
        return librosa.feature.chroma_stft(y=y, sr=self.sr)

    # -----------------------------------------------------
    # SAVE PNG
    # -----------------------------------------------------
    def save_png_feature(self, feature, out_path):
        plt.figure(figsize=(3, 3))
        plt.axis("off")
        plt.imshow(feature, aspect="auto", origin="lower")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def save_rgb_feature(self, mel, mfcc, chroma, out_path):
        """Combine mel, mfcc, chroma into a single RGB image and save as PNG.

        The three feature maps are normalized independently and resized to
        `(self.n_mels, self.n_mels)` by default (square image). If `save_png` is
        False, this method will save a single numpy array with shape (H, W, 3).
        """
        # target size: width, height
        target = (self.n_mels, self.n_mels)

        r = _normalize_and_resize(mel, target)
        g = _normalize_and_resize(mfcc, target)
        b = _normalize_and_resize(chroma, target)

        rgb = np.stack([r, g, b], axis=2)

        if self.save_png:
            Image.fromarray(rgb).save(out_path)
        else:
            np.save(out_path, rgb)

    # -----------------------------------------------------
    # PROCESS SINGLE FILE
    # -----------------------------------------------------
    def process_file(self, audio_path):
        y = self.load_audio(audio_path)
        if y is None:
            return False

        try:
            mel = self.extract_mel(y)
            mfcc = self.extract_mfcc(y)
            chroma = self.extract_chroma(y)
        except Exception as e:
            print(f"[WARN] Error processing '{audio_path}': {e}")
            return False

        base = Path(audio_path).stem

        if self.save_png:
            # Save combined 3-channel image
            self.save_rgb_feature(mel, mfcc, chroma, self.output_dir / f"{base}_rgb.png")
        else:
            np.save(self.output_dir / f"{base}_features.npy", np.stack([mel, mfcc, chroma], axis=2))

        return True

    # -----------------------------------------------------
    # PROCESS DATASET
    # -----------------------------------------------------
    def process_dataset(self, raw_dataset_path, output_path=None):
        raw_dataset_path = os.path.abspath(raw_dataset_path)
        out_dir = Path(output_path) if output_path else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

        processed = 0
        skipped = 0
        skipped_files = []

        prev_out = self.output_dir
        self.output_dir = out_dir

        for root, _, files in os.walk(raw_dataset_path):
            for fname in files:
                if Path(fname).suffix.lower() in exts:
                    fpath = os.path.join(root, fname)

                    ok = self.process_file(fpath)
                    if ok:
                        processed += 1
                    else:
                        skipped += 1
                        skipped_files.append(fpath)

                    # augmentation
                    if self.augment and ok:
                        try:
                            y = self.load_audio(fpath)
                            y_aug = augment_audio(y, self.sr)

                            mel = self.extract_mel(y_aug)
                            mfcc = self.extract_mfcc(y_aug)
                            chroma = self.extract_chroma(y_aug)

                            base = Path(fpath).stem
                            if self.save_png:
                                self.save_rgb_feature(mel, mfcc, chroma, out_dir / f"{base}_aug_rgb.png")
                            else:
                                np.save(out_dir / f"{base}_aug_features.npy", np.stack([mel, mfcc, chroma], axis=2))

                        except Exception as e:
                            print(f"[WARN] Augmentation failed for '{fpath}': {e}")

        self.output_dir = prev_out

        # record skipped files
        if skipped_files:
            with open(out_dir / "skipped_files.txt", "w") as f:
                f.write("\n".join(skipped_files))

        print(f"[INFO] Completed. Processed: {processed}, Skipped: {skipped}")
        return processed
