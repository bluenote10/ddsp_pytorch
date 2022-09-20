from os import path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import typer
import yaml
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ddsp.core import mean_std_loudness, multiscale_fft, safe_log
from ddsp.model import DDSP
from ddsp.utils import get_scheduler
from preprocess import Dataset


def train(
    config_name: str = "config.yaml",
    name: str = "debug",
    root: str = "runs",
    data_dir: str = "data_dir",
    steps: int = 500000,
    batch_size: int = 16,
    start_lr: float = 1e-3,
    stop_lr: float = 1e-4,
    decay_over: int = 400000,
):

    with open(config_name, "r") as f:
        config: Any = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DDSP(**config["model"]).to(device)

    # dataset = Dataset(config["preprocess"]["out_dir"])
    dataset = Dataset(data_dir)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    mean_loudness, std_loudness = mean_std_loudness(dataloader)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness

    writer = SummaryWriter(path.join(root, name), flush_secs=20)

    with open(path.join(root, name, "config.yaml"), "w") as out_config:
        yaml.safe_dump(config, out_config)

    opt = torch.optim.Adam(model.parameters(), lr=start_lr)

    schedule = get_scheduler(
        len(dataloader),
        start_lr,
        stop_lr,
        decay_over,
    )

    # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

    scales = config["train"]["scales"]
    overlap = config["train"]["overlap"]

    best_loss = float("inf")
    mean_loss = 0.0
    n_element = 0
    step = 0
    epochs = int(np.ceil(steps / len(dataloader)))

    for e in tqdm(range(epochs)):
        print(f"epoch: {e+1}")

        for s, p, l in dataloader:
            s = s.to(device)
            p = p.unsqueeze(-1).to(device)
            l = l.unsqueeze(-1).to(device)

            l = (l - mean_loudness) / std_loudness

            y = model(p, l).squeeze(-1)

            ori_stft = multiscale_fft(s, scales, overlap)
            rec_stft = multiscale_fft(y, scales, overlap)

            loss = torch.tensor(0.0).to(device)
            for s_x, s_y in zip(ori_stft, rec_stft):
                lin_loss = (s_x - s_y).abs().mean()
                log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
                loss += lin_loss + log_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            writer.add_scalar("loss", loss.item(), step)

            step += 1

            n_element += 1
            mean_loss += (loss.item() - mean_loss) / n_element

            print(f"mean_loss = {mean_loss}")

        if e % 10 == 0:
            writer.add_scalar("lr", schedule(e), e)
            writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
            writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)
            # scheduler.step()
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(
                    model.state_dict(),
                    path.join(root, name, "state.pth"),
                )

            mean_loss = 0
            n_element = 0

            audio = torch.cat([s, y], -1).reshape(-1).detach().cpu().numpy()

            sf.write(
                path.join(root, name, f"eval_{e:06d}.wav"),
                audio,
                config["preprocess"]["sampling_rate"],
            )


def main() -> None:
    app = typer.Typer()
    app.command()(train)
    app()


if __name__ == "__main__":
    main()
