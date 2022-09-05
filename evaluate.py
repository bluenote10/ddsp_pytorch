from pathlib import Path

import numpy as np
import soundfile
import torch

model = torch.jit.load("export/ddsp_debug_pretrained.ts")

input_dir = Path("/home/fabian/gdrive/colab/ddsp_pytorch/input/ddsp_preprocessed")


def write_audio(pitch: torch.Tensor, loudness: torch.Tensor, file: Path):
    audio = model(pitch, loudness)
    audio_raw = audio.detach().numpy().reshape(-1)
    soundfile.write(file, audio_raw, samplerate=16000)  # , subtype='PCM_24')


if True:
    freq = 500.0
    pitch = torch.ones(1, 200, 1) * freq

    loudness = torch.tensor(
        np.concatenate([np.linspace(-8.5, -3.0, 5), np.linspace(-3.0, -8.5, 195)]),
        dtype=torch.float32,
    ).reshape(1, -1, 1)

    write_audio(pitch, loudness, f"/tmp/synth_{freq}.wav")


if False:
    pitches = np.load(input_dir / "pitchs.npy")
    loudnesses = np.load(input_dir / "loudness.npy")

    for i in range(len(pitches)):
        print(f"Processing: {i}")

        pitch = torch.tensor(pitches[i]).reshape(1, -1, 1)
        loudness = torch.tensor(loudnesses[i]).reshape(1, -1, 1)

        write_audio(pitch, loudness, f"/tmp/ddsp_{i:05d}.wav")
