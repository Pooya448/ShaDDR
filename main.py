import argparse
import wandb
from configs.config import load_config
from trainer_tmp import TrainWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decorgan")
    parser.add_argument("--config", "-c", type=str, help="Path to config file.")
    parser.add_argument("--task", "-t", type=str, help="The task to execute.")
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        default=False,
        help="Whether this is a debug run.",
    )
    args = parser.parse_args()

    config = load_config(args.config, None)
    wrapper = TrainWrapper(opt=config)

    if args.debug or args.task == "test":
        wandb.init(mode="disabled")

    if args.task == "train":
        wrapper.train()
    elif args.task == "test":
        wrapper.test()
    elif args.task == "eval_fid":
        wrapper.eval_fid()
    elif args.task == "train_fid":
        wrapper.train_fid()
    elif args.task == "train_cls":
        wrapper.train_cls()
    elif args.task == "eval_cls":
        wrapper.eval_cls()
    else:
        raise Exception("Invalid Task!")
