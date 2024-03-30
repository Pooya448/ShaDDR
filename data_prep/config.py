import json
import os
from dataclasses import dataclass
from datetime import datetime

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config:
    workers: int = 8  # check how many CPU's you have available! number of workers for obj --> SDF
    resolution: int = 256  # 320  # 320  # 384        #256  # sdf voxel res
    filetype: str = "voxel"
    debug_mode: bool = False  # process 1 mesh at the time + verbose
    data_folder: str = r"data/raw/house"
    save_folder: str = r"data/preprocessed/house_feb_26"  # output Voxels
    creation_date: str = datetime.now().strftime("%m-%d_%H-%M-%S")


def save_config(config):
    jsonconfig = json.dumps(json.loads(config.to_json()), indent=4)
    with open(os.path.join(config.save_folder, "config.json"), "w") as f:
        f.write(jsonconfig)
