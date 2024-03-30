import os
from datetime import datetime
import pytz
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from glob import glob
from dataloader.SDFDataModule import SDFDataModule
from dataloader.VoxelDataModule import VoxelDataModule
from model.DecorGAN import DecorGAN


class TrainWrapper(object):
    def __init__(self, opt):
        torch.set_float32_matmul_precision("medium")
        self.opt = opt

        self.data_module = VoxelDataModule(opt["data"], batch_size=opt["train"]["batch_size"])

    def train(self):
        tz = pytz.timezone("America/Vancouver")
        creation_date = datetime.now(tz).strftime("%m-%d_%H-%M")

        run_folder = f"{creation_date}_{self.opt['experiment']['data_rep']}_{self.opt['log']['run_name']}"
        run_folder = os.path.join("runs/", run_folder)

        checkpoint_dir = os.path.join(run_folder, "checkpoints")
        val_output_dir = os.path.join(run_folder, "val_output")
        test_output_dir = os.path.join(run_folder, "test_output")

        for d in [checkpoint_dir, val_output_dir, test_output_dir]:
            os.makedirs(d, 0o777, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir)

        opt_log = {"val_log_dir": val_output_dir, "test_log_dir": test_output_dir}

        train_model = DecorGAN(self.opt["model"], self.opt["train"], opt_log, self.opt["experiment"]["data_rep"])
        wandb_logger = WandbLogger(
            name="{}_{}_{}".format("train", self.opt["experiment"]["data_rep"], self.opt["log"]["run_name"]),
            project=self.opt["log"]["project"],
            log_model=self.opt["log"]["log_model"],
        )
        wandb_logger.experiment.config.update(self.opt)
        trainer = L.Trainer(
            logger=wandb_logger,
            log_every_n_steps=1,
            check_val_every_n_epoch=self.opt["train"]["val_freq"],
            callbacks=[checkpoint_callback],
            accelerator=self.opt["system"]["accelerator"],
            devices=self.opt["system"]["n_gpu"],
            precision=self.opt["system"]["precision"],
            max_epochs=self.opt["train"]["epochs"],
            fast_dev_run=self.opt["debug"]["fast_dev_run"],
        )

        trainer.fit(model=train_model, datamodule=self.data_module)

    def test(self):
        run_name = self.opt["test"]["run_name"]
        ckpt_pattern = os.path.join("runs/", run_name, "checkpoints", "*.ckpt")
        ckpt_file = glob(ckpt_pattern)[0]

        val_output_dir = os.path.join("runs/", run_name, "val_output")
        test_output_dir = os.path.join("runs/", run_name, "test_output")
        opt_log = {"val_log_dir": val_output_dir, "test_log_dir": test_output_dir}

        test_model = DecorGAN.load_from_checkpoint(
            ckpt_file,
            opt=self.opt["model"],
            opt_train=self.opt["train"],
            opt_log=opt_log,
            data_rep=self.opt["experiment"]["data_rep"],
        )

        trainer = L.Trainer(
            accelerator=self.opt["system"]["accelerator"],
            devices=self.opt["system"]["n_gpu"],
            precision=self.opt["system"]["precision"],
        )
        trainer.test(model=test_model, datamodule=self.data_module)

    def eval_fid(self):
        pass

    def train_fid(self):
        pass

    def train_cls(self):
        pass

    def eval_cls(self):
        pass
