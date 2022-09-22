from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf  # noqa
import torch
import typer
import yaml
from tqdm import tqdm

from ddsp import io
from ddsp.core import extract_loudness, extract_pitch

"""
python preprocess.py --base-path ~/Dropbox/Temp/ReferenceAudio/Landola/Snapshot1/ --glob-pattern recording_normalized.mp3
"""


def get_files(base_path: Path, glob_pattern: str):
    return list(base_path.rglob(glob_pattern))


def preprocess(f, sampling_rate, block_size, signal_length, oneshot):
    x, sr = librosa.load(f, sr=sampling_rate)
    assert sr == sampling_rate

    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir: Path):
        super().__init__()
        self.signals = np.load(out_dir / "signals.npy")
        self.pitchs = np.load(out_dir / "pitchs.npy")
        self.loudness = np.load(out_dir / "loudness.npy")

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])
        return s, p, l


def preprocess_main(
    base_path: Path = typer.Option(...), glob_pattern: str = typer.Option(...)
):
    with open(Path(__file__).parent / "config.yaml", "r") as path:
        config = yaml.safe_load(path)

    files = get_files(base_path, glob_pattern)
    assert len(files) > 0, "No files found"

    out_dir = io.get_data_dir()
    out_dir.mkdir(exist_ok=True, parents=True)

    signals = []
    pitchs = []
    loudness = []

    progress_bar = tqdm(files)
    for path in progress_bar:
        progress_bar.set_description(str(path))

        x, p, l = preprocess(path, **config["preprocess"])
        signals.append(x)
        pitchs.append(p)
        loudness.append(l)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    np.save(out_dir / "signals.npy", signals)
    np.save(out_dir / "pitchs.npy", pitchs)
    np.save(out_dir / "loudness.npy", loudness)


def main() -> None:
    app = typer.Typer()
    app.command()(preprocess_main)
    app()


if __name__ == "__main__":
    main()
