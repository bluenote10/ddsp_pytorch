from pathlib import Path

import numpy as np
import soundfile
import torch
from torchviz import make_dot


def visualize_model(
    model: torch.nn.Module, pitch: torch.Tensor, loudness: torch.Tensor
):
    # It looks like torchviz requires inputs to actually visualize something...
    y = model(pitch, loudness)

    graph = make_dot(y.mean(), params=dict(list(model.named_parameters())))
    graph.render("/tmp/ddsp_model.png", format="png")


def inspect_model(model):
    print("Model summary:")
    print(f"block_size: {model.ddsp.block_size}")
    print(f"sample_rate: {model.ddsp.sampling_rate}")

    def stringify(x):
        if isinstance(x, torch.Tensor):
            return f"Tensor of shape {tuple(x.shape)}"
        else:
            return f"Instance of {type(x).__name__}"

    print("Modules:")
    for name, module in model.named_modules():
        print(f" - {name}: {stringify(module)}")

    print("Children:")
    for name, child in model.named_children():
        print(f" - {name}: {stringify(child)}")

    print("Parameters:")
    for name, param in model.named_parameters():
        print(f" - {name}: {stringify(param)}")

    print("Buffers:")
    for name, buffer in model.named_buffers():
        print(f" - {name}: {stringify(buffer)}")


def write_audio(model, pitch: torch.Tensor, loudness: torch.Tensor, file: Path):
    assert pitch.shape[-2] == loudness.shape[-2]
    input_length = pitch.shape[-2]

    audio = model(pitch, loudness)
    audio_raw = audio.detach().numpy().reshape(-1)

    output_length = len(audio_raw)
    ratio = output_length / input_length
    print(
        f"input_length = {input_length}, output_length = {output_length}, ratio = {ratio}"
    )

    soundfile.write(file, audio_raw, samplerate=16000)  # , subtype='PCM_24')


def main():
    model = torch.jit.load("export/ddsp_debug_pretrained.ts")
    assert isinstance(model, torch.nn.Module)

    inspect_model(model)

    block_size = model.ddsp.block_size
    sample_rate = model.ddsp.sampling_rate
    print(f"Model step size: {block_size / sample_rate * 1000} ms")

    if True:
        freq = 500.0
        pitch = torch.ones(1, 200, 1) * freq

        loudness = torch.tensor(
            np.concatenate([np.linspace(-8.5, -3.0, 5), np.linspace(-3.0, -8.5, 195)]),
            dtype=torch.float32,
        ).reshape(1, -1, 1)

        write_audio(model, pitch, loudness, f"/tmp/synth_{freq}.wav")

        print("Visualizing model...")
        visualize_model(model, pitch, loudness)

    if False:
        input_dir = Path(
            "/home/fabian/gdrive/colab/ddsp_pytorch/input/ddsp_preprocessed"
        )

        pitches = np.load(input_dir / "pitchs.npy")
        loudnesses = np.load(input_dir / "loudness.npy")

        for i in range(len(pitches)):
            print(f"Processing: {i}")

            pitch = torch.tensor(pitches[i]).reshape(1, -1, 1)
            loudness = torch.tensor(loudnesses[i]).reshape(1, -1, 1)

            write_audio(model, pitch, loudness, f"/tmp/ddsp_{i:05d}.wav")


if __name__ == "__main__":
    main()
