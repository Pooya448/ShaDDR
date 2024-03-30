import glob
import os
import sys
from dataclasses import dataclass
from multiprocessing import Pool
import numpy as np
from config import Config, save_config

# local imports
from tqdm import tqdm
import binvox_rw

sys.path.insert(0, "/home/ubuntu/webhooks")


class Binvox2NpzPrep(object):
    def __init__(self, config: dataclass):
        self.config = config

    def convert(self, save_folder):
        """converts all binvox in data_folder to npz in save_folder."""
        bin_paths = self.make_input_list(self.config.data_folder, self.config.filetype)

        print("------------------")
        print(f"STARTING PREPROCESSING OF {len(bin_paths)} FILES.")
        print("------------------")

        self.convert_to_npz(bin_paths, self.config.resolution, save_folder, self.config)

        print("NPZ GENERATION COMPLETE.")

    def make_input_list(self, data_folder, filetype):
        """makes list of all binvox files to be processed"""
        pattern = f"{data_folder}/*/*.binvox"
        bin_paths = glob.glob(pattern, recursive=True)

        bin_paths.sort()
        bin_paths = sorted([item.strip().split("/") for item in bin_paths])
        assert len(bin_paths) != 0, "No objs found. check data_folder"
        return bin_paths

    def convert_to_npz(
        self,
        bin_paths,
        resolution,
        save_folder,
        config,
    ):
        """converts a single binvox to Npz"""

        # make list of all files to process
        args = self.make_file_list(bin_paths, save_folder, resolution, config)
        total_files = len(args)

        # process files
        if config.debug_mode:
            for arg in args:
                self.to_npz(arg)
        else:
            with Pool(processes=config.workers) as pool, tqdm(total=total_files) as pbar:
                for _ in pool.imap_unordered(self.to_npz, args):
                    pbar.update()

    def to_npz(self, arg):
        (idx, bin_path, save_path, resolution) = arg

        vox_256 = self.get_vox_from_binvox(bin_path)
        np.savez_compressed(save_path, voxel=vox_256)

    def make_file_list(self, bin_list, save_folder, resolution, config: dataclass):
        """makes list of all files to be processed from binvox to npz. format for multiprocessing"""

        args = []
        c = 0
        for idx, bins in enumerate(bin_list):
            save_file = os.path.join(save_folder, f"{bins[-3]}_{idx}" + f".{config.filetype}_{resolution}.npz")
            if not os.path.exists(save_file):
                arg = (
                    idx,
                    "/".join(bins),
                    save_file,
                    resolution,
                )
                args.append(arg)
            else:
                c += 1
        print(f"num already done is {c}")
        print(f"{len(args)} left to be processed!")
        return args

    def get_vox_from_binvox(self, objname):
        # get voxel models
        voxel_model_file = open(objname, "rb")
        voxel_model_512 = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=True).data.astype(np.uint8)
        step_size = 2
        voxel_model_256 = voxel_model_512[0::step_size, 0::step_size, 0::step_size]
        for i in range(step_size):
            for j in range(step_size):
                for k in range(step_size):
                    voxel_model_256 = np.maximum(
                        voxel_model_256, voxel_model_512[i::step_size, j::step_size, k::step_size]
                    )
        # add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
        # voxel_model_256 = np.flip(np.transpose(voxel_model_256, (2,1,0)),2)

        return voxel_model_256


if __name__ == "__main__":
    # use dataclass as config
    config = Config()

    os.makedirs(config.save_folder, exist_ok=True)

    # save the config to the save folder, pretty printed
    save_config(config)

    save_folders = {"voxel": os.path.join(config.save_folder, "voxel")}

    # write the folders
    for folder in save_folders.values():
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)

    vox2npz = Binvox2NpzPrep(config)
    vox2npz.convert(save_folders["voxel"])
