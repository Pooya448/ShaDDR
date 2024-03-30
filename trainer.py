import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from models.Generator import Generator
from models.Discriminator3D import Discriminator3D
from models.Discriminator2D import Discriminator2D


class Trainer:
    def __init__(self, config):
        """
        Initializes the trainer with the dataset, model, optimizer, and device.

        Parameters:
            dataset (torch.utils.data.Dataset): The dataset for training.
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for training the model.
            device (str, optional): The device to train on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        self.mask_margin = 16
        self.save_epoch = 1
        self.sampling_threshold = 0.4
        self.render_view_id = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.voxel_renderer = voxel_renderer(self.real_size) #TODO: handle voxel renderer

        # Model dimensions
        self.in_size = config["in_size"]
        self.out_size = config["out_size"]
        self.dim_g = config["dim_g"]
        self.dim_d = config["dim_d"]
        self.dim_z = config["dim_z"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]

        self.up_factor = self.output_size // self.input_size

        self.real_size = config.output_size  # TODO: Change with self.output_size
        self.asymmetry = True  # TODO: always true, remove

        self.run_folder = config["run_folder"]
        self.train_mode = config["train_mode"]  # 'geometry' or 'texture'

        self.n_style = 8

        self.create_models_and_optimizers()

    def create_models_and_optimizers(self):

        self.generator = self.create_generators()
        self.optim_g = Adam(self.generator.parameters(), lr=0.0001)

        if self.train_mode == "geometry":
            self.geo_d_l, self.geo_d_s = self.create_geo_discriminators()
            self.optim_d_l = Adam(self.geo_d_l.parameters(), lr=0.0001)
            self.optim_d_s = Adam(self.geo_d_s.parameters(), lr=0.0001)

        elif self.train_mode == "texture":
            self.tex_ds = self.create_tex_discriminators()
            params_to_train = [x.parameters() for x in self.tex_ds]
            self.optim_d_tex = Adam(params_to_train, lr=0.0001)

    def create_generators(self):
        return Generator(dim_g=self.dim_g, dim_z=self.dim_z, dim_prob=self.n_style).to(
            self.device
        )

    def create_geo_discriminators(self):
        geo_d_l = Discriminator3D(dim_d=self.dim_d, dim_z=self.n_style + 1).to(
            self.device
        )
        geo_d_s = Discriminator3D(dim_d=self.dim_d, dim_z=self.n_style + 1).to(
            self.device
        )
        return geo_d_l, geo_d_s

    def create_tex_discriminators(self):
        self.tex_d_back = Discriminator2D(
            dim_d=self.dim_d, dim_z=self.n_style + 1, d_in=4
        ).to(self.device)
        self.tex_d_front = Discriminator2D(
            dim_d=self.dim_d, dim_z=self.n_style + 1, d_in=4
        ).to(self.device)
        self.tex_d_top = Discriminator2D(
            dim_d=self.dim_d, dim_z=self.n_style + 1, d_in=4
        ).to(self.device)
        self.tex_d_side = Discriminator2D(
            dim_d=self.dim_d, dim_z=self.n_style + 1, d_in=4
        ).to(self.device)
        self.tex_d_right = Discriminator2D(
            dim_d=self.dim_d, dim_z=self.n_style + 1, d_in=4
        ).to(self.device)
        tex_ds = [
            self.tex_d_back,
            self.tex_d_front,
            self.tex_d_top,
            self.tex_d_side,
            self.tex_d_right,
        ]
        return tex_ds
        params_to_train = [x.parameters() for x in tex_ds]
        self.optim_d_tex = Adam(params_to_train, lr=0.0001)

    def train(self, epochs, is_geometry_training=True):

        pass

    def train_geometry(self, epochs):

        style_dataset = None  # TODO: implement StyleDataset
        dataloder = None  # TODO: implement DataLoader from Dataset
        for epoch in range(epochs):

            self.geo_d_s.train()
            self.geo_d_l.train()
            self.generator.train()

            for i, batch in enumerate(dataloder):
                z_vector_geometry = np.zeros([self.n_style], np.float32)
                z_vector_geometry_idx = np.random.randint(self.n_style)
                z_vector_geometry[z_vector_geometry_idx] = 1
                z_geometry_tensor = (
                    torch.from_numpy(z_vector_geometry).to(self.device).view([1, -1])
                )

    def train_texture(self, voxels, geometry_z, texture_z, mask_):
        pass

    def compute_loss(self, output):
        pass

    def post_epoch_actions(self, epoch):
        pass
