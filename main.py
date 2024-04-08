import torch

import wandb
import pytz
from datetime import datetime
import argparse
import shutil
import sys

from pathlib import Path

from trainer import ShaddrTrainer as Trainer
from configs.config import load_configuration
import torch.multiprocessing as mp

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method("spawn", force=True)
    # torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")

    tz = pytz.timezone("America/Vancouver")
    creation_date = datetime.now(tz).strftime("%m-%d_%H-%M")

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./configs/shaddr.yaml")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--run", type=str, default="unnamed")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    config = load_configuration(args.conf)

    run_ = args.run
    run = None
    if args.mode == "train":
        if Path(run_).exists():
            if args.use_checkpoint:
                run_folder = Path(run_)
                wandb_run_name = "_".join(run_folder.name.split("_")[1:])
                print("Run exists. Resuming train on existing run.")
                run = wandb.init(
                    name=wandb_run_name, project="ShaDDR", config=config, resume=True
                )
            else:
                raise ValueError(
                    "Run already exists. Please provide a new run name. (or use --use_checkpoint to resume existing run)"
                )
        else:
            print("Creating new run.")
            run_name = f"{creation_date}_{args.run}"
            wandb_run_name = f"{args.run}_chair"
            run_folder = Path(config["base_dir"]) / run_name
            run_folder.mkdir(parents=True, exist_ok=True)
            if args.debug:
                wandb.init(mode="disabled")
            else:
                run = wandb.init(name=wandb_run_name, project="ShaDDR", config=config)
                run.log_code()

    elif args.mode == "test":
        if Path(run_).exists():
            print("Run exists. Testing on existing run.")
            run_folder = Path(run_)
        else:
            raise ValueError("Run does not exist. Please provide a valid run path.")

    trainer = Trainer(config, run_folder)
    t = config["train_mode"]
    if t == "geometry":
        trainer.train_geometry(epochs=10)
    elif t == "texture":
        trainer.train_texture(epochs=10)
