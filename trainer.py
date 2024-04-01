import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from pathlib import Path

from models.Generator import Generator
from models.Discriminator3D import Discriminator3D
from models.Discriminator2D import Discriminator2D

from dataloader.ShaddrData import ShaddrDataset
from utils import render_voxel, get_tex_mask_d
from tqdm import tqdm


class ShaddrTrainer:
    def __init__(self, config, run_folder):
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
        self.threshold = 0.4
        self.render_view_id = 0
        self.d_steps = 1
        self.r_steps = 1
        self.g_steps = 1
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
        self.delta = 0.1
        self.gamma = 1

        self.up_factor = self.out_size // self.in_size

        # self.real_size = config.out_size  # TODO: Change with self.output_size

        self.run_folder = run_folder
        self.train_mode = config["train_mode"]  # 'geometry' or 'texture'
        self.data_dir = config["data_dir"]

        self.n_style = 4

        self.create_models_and_optimizers()

        train_split = Path("splits/content_train.txt")
        test_split = Path("splits/content_test.txt")
        style_split = Path("splits/style.txt")

        self.style_dataset = ShaddrDataset(
            self.data_dir, is_style=True, split_file=style_split
        )
        self.train_dataset = ShaddrDataset(
            self.data_dir, is_style=False, split_file=train_split
        )
        self.test_dataset = ShaddrDataset(
            self.data_dir, is_style=False, split_file=test_split
        )

    def create_datasets(self, rp, is_style, split_file):
        return ShaddrDataset(rp, is_style, split_file)

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
        tex_ds = [
            self.tex_d_back,
            self.tex_d_front,
            self.tex_d_top,
            self.tex_d_side,
        ]
        return tex_ds
        params_to_train = [x.parameters() for x in tex_ds]
        self.optim_d_tex = Adam(params_to_train, lr=0.0001)

    def train(self, epochs, is_geometry_training=True):

        pass

    def train_geometry(self, epochs):
        cnt_loader = DataLoader(
            self.train_dataset, batch_size=1, shuffle=True, num_workers=24
        )
        # stl_loader = DataLoader(self.style_dataset, batch_size=1, shuffle=True)

        for epoch in tqdm(range(epochs)):

            self.geo_d_s.train()
            self.geo_d_l.train()
            self.generator.train()

            for i, batch in enumerate(cnt_loader):

                # z_ = np.zeros([self.n_style], np.float32)
                # z_idx = np.random.randint(self.n_style)
                # z_[z_idx] = 1
                # z_torch = torch.from_numpy(z_).to(self.device).view([1, -1])

                z_torch, z_idx = self.sample_random_style()

                # style_batch = self.style_dataset.__getitem__(z_idx)
                # mask_g =
                cnt_mask_g = batch["mask_g"].to(self.device).unsqueeze(0).float()
                cnt_mask_d_l = batch["mask_d_l"].to(self.device).unsqueeze(0).float()
                cnt_mask_d_s = batch["mask_d_s"].to(self.device).unsqueeze(0).float()
                cnt_geo_in = batch["geo_in"].to(self.device).unsqueeze(0).float()

                geo_code = torch.matmul(z_torch, self.generator.geometry_codes).view(
                    [1, -1, 1, 1, 1]
                )
                voxel_gen_l, voxel_gen_s = self.generator(
                    voxels=cnt_geo_in,
                    geometry_z=geo_code,
                    texture_z=None,
                    mask=cnt_mask_g,
                    is_geometry_training=True,
                )
                voxel_gen_s = voxel_gen_s.detach()
                voxel_gen_l = voxel_gen_l.detach()

                ### Discriminator training
                for _ in range(self.d_steps):

                    self.geo_d_l.zero_grad()
                    self.geo_d_s.zero_grad()

                    style_batch = self.style_dataset.__getitem__(z_idx)
                    style_mask_d_l = (
                        style_batch["mask_d_l"].to(self.device).unsqueeze(0).float()
                    )
                    style_mask_d_s = (
                        style_batch["mask_d_s"].to(self.device).unsqueeze(0).float()
                    )
                    style_geo_l = (
                        style_batch["geo_l"].to(self.device).unsqueeze(0).float()
                    )
                    style_geo_s = (
                        style_batch["geo_s"].to(self.device).unsqueeze(0).float()
                    )

                    # Large -> 256
                    d_l = self.geo_d_l(style_geo_l, is_training=True)

                    loss_real_l = self.calculate_d_loss(
                        d_l, style_mask_d_l, z_idx, is_real=True
                    )
                    loss_real_l.backward()

                    d_l = self.geo_d_l(voxel_gen_l, is_training=True)
                    loss_fake_l = self.calculate_d_loss(
                        d_l, cnt_mask_d_l, z_idx, is_real=False
                    )
                    loss_fake_l.backward()

                    self.optim_d_l.step()
                    self.geo_d_l.zero_grad()

                    # Small
                    d_s = self.geo_d_s(style_geo_s, is_training=True)
                    loss_real_s = self.calculate_d_loss(
                        d_s, style_mask_d_s, z_idx, is_real=True
                    )

                    loss_real_s.backward()

                    d_s = self.geo_d_s(voxel_gen_s, is_training=True)
                    loss_fake_s = self.calculate_d_loss(
                        d_s, cnt_mask_d_s, z_idx, is_real=False
                    )
                    loss_fake_s.backward()

                    self.optim_d_s.step()
                    self.geo_d_s.zero_grad()

                ### Reconstruction training
                for _ in range(self.r_steps):
                    z_torch_r, z_idx_r = self.sample_random_style()

                    style_batch_r = self.style_dataset.__getitem__(z_idx_r)
                    style_geo_l = (
                        style_batch_r["geo_l"].to(self.device).unsqueeze(0).float()
                    )
                    style_geo_s = (
                        style_batch_r["geo_s"].to(self.device).unsqueeze(0).float()
                    )
                    style_mask_g = (
                        style_batch_r["mask_g"].to(self.device).unsqueeze(0).float()
                    )
                    style_geo_in = (
                        style_batch_r["geo_in"].to(self.device).unsqueeze(0).float()
                    )

                    self.generator.zero_grad()

                    geo_code_r = torch.matmul(
                        z_torch_r, self.generator.geometry_codes
                    ).view([1, -1, 1, 1, 1])

                    recon_gen_l, recon_gen_s = self.generator(
                        voxels=style_geo_in,
                        geometry_z=geo_code_r,
                        texture_z=None,
                        mask=style_mask_g,
                        is_geometry_training=True,
                    )
                    loss_r = self.calculate_r_loss(
                        gen_s=recon_gen_s,
                        gen_l=recon_gen_l,
                        gt_s=style_geo_s,
                        gt_l=style_geo_l,
                        beta=self.beta,
                    )

                    loss_r.backward()
                    self.optim_g.step()
                    self.generator.zero_grad()

                ### Generator training
                for _ in range(self.g_steps):
                    self.generator.zero_grad()
                    geo_code = torch.matmul(
                        z_torch, self.generator.geometry_codes
                    ).view([1, -1, 1, 1, 1])
                    voxel_gen_l, voxel_gen_s = self.generator(
                        voxels=cnt_geo_in,
                        geometry_z=geo_code,
                        texture_z=None,
                        mask=cnt_mask_g,
                        is_geometry_training=True,
                    )
                    d_l = self.geo_d_l(voxel_gen_l, is_training=True)
                    d_s = self.geo_d_s(voxel_gen_s, is_training=True)

                    loss_g_l = self.calculate_g_loss(
                        d_l, cnt_mask_d_l, z_idx, self.alpha
                    )
                    loss_g_s = self.calculate_g_loss(
                        d_s, cnt_mask_d_s, z_idx, self.alpha
                    )
                    loss_g = loss_g_s * self.delta + loss_g_l * self.gamma

                    loss_g.backward()
                    self.optim_g.step()
                    self.generator.zero_grad()

                    report_str = f"Epoch: {epoch}, loss_g: {loss_g.item()}, loss_r: {loss_r.item()}, loss_d: {loss_fake_l.item() + loss_fake_s.item()}"
                    print(report_str)

            report_str = f"Epoch: {epoch}, loss_g: {loss_g.item()}, loss_r: {loss_r.item()}, loss_d: {loss_fake_l.item() + loss_fake_s.item()}"
            print(report_str)

    def train_texture(self, epochs):
        cnt_loader = DataLoader(
            self.train_dataset, batch_size=1, shuffle=True, num_workers=24
        )
        # stl_loader = DataLoader(self.style_dataset, batch_size=1, shuffle=True)

        for epoch in range(epochs):

            self.tex_d_back.train()
            self.tex_d_front.train()
            self.tex_d_top.train()
            self.tex_d_side.train()

            self.generator.train()

            for i, batch in enumerate(cnt_loader):

                # z_torch_geo, z_idx_geo = self.sample_random_style()
                # z_torch_tex, z_idx_tex = self.sample_random_style()

                z_torch_geo, z_torch_tex, z_style_idx = self.sample_style_tex()

                cnt_mask_g = batch["mask_g"].to(self.device).unsqueeze(0).float()
                # cnt_mask_d_l = batch["mask_d_l"].to(self.device).unsqueeze(0).float()
                # cnt_mask_d_s = batch["mask_d_s"].to(self.device).unsqueeze(0).float()
                cnt_geo_in = batch["geo_in"].to(self.device).unsqueeze(0).float()

                geo_code = torch.matmul(
                    z_torch_geo, self.generator.geometry_codes
                ).view([1, -1, 1, 1, 1])
                tex_code = torch.matmul(z_torch_tex, self.generator.texture_codes).view(
                    [1, -1, 1, 1, 1]
                )

                geo_gen, tex_gen = self.generator(
                    voxels=cnt_geo_in,
                    geometry_z=geo_code,
                    texture_z=tex_code,
                    mask=cnt_mask_g,
                    is_geometry_training=False,
                )
                geo_gen = geo_gen.detach()
                tex_gen = tex_gen.detach()

                back_tex, front_tex, top_tex, side_tex = render_voxel(
                    (geo_gen > self.threshold).float(), tex_gen
                )
                mask_d_back, mask_d_front, mask_d_top, mask_d_side = get_tex_mask_d(
                    [back_tex, front_tex, top_tex, side_tex]
                )

                ### Discriminator training
                for _ in range(self.d_steps):

                    self.tex_d_back.zero_grad()
                    self.tex_d_front.zero_grad()
                    self.tex_d_top.zero_grad()
                    self.tex_d_side.zero_grad()

                    style_batch = self.style_dataset.__getitem__(z_style_idx)
                    renders_list = style_batch["renders"].to(self.device)

                    style_back_tex, style_front_tex, style_top_tex, style_side_tex = (
                        renders_list[0].permute(2, 0, 1).unsqueeze(0).float(),
                        renders_list[1].permute(2, 0, 1).unsqueeze(0).float(),
                        renders_list[2].permute(2, 0, 1).unsqueeze(0).float(),
                        renders_list[3].permute(2, 0, 1).unsqueeze(0).float(),
                    )
                    (
                        style_mask_d_back,
                        style_mask_d_front,
                        style_mask_d_top,
                        style_mask_d_side,
                    ) = get_tex_mask_d(
                        [style_back_tex, style_front_tex, style_top_tex, style_side_tex]
                    )

                    d_back = self.tex_d_back(style_back_tex, is_training=True)
                    d_front = self.tex_d_front(style_front_tex, is_training=True)
                    d_top = self.tex_d_top(style_top_tex, is_training=True)
                    d_side = self.tex_d_side(style_side_tex, is_training=True)

                    loss_real_back = self.calculate_d_loss(
                        d_back, style_mask_d_back, z_style_idx, is_real=True
                    )
                    loss_real_front = self.calculate_d_loss(
                        d_front, style_mask_d_front, z_style_idx, is_real=True
                    )
                    loss_real_top = self.calculate_d_loss(
                        d_top, style_mask_d_top, z_style_idx, is_real=True
                    )
                    loss_real_side = self.calculate_d_loss(
                        d_side, style_mask_d_side, z_style_idx, is_real=True
                    )

                    total_style = (
                        loss_real_back
                        + loss_real_front
                        + loss_real_top
                        + loss_real_side
                    )
                    total_style.backward()

                    d_back = self.tex_d_back(back_tex, is_training=True)
                    d_front = self.tex_d_front(front_tex, is_training=True)
                    d_top = self.tex_d_top(top_tex, is_training=True)
                    d_side = self.tex_d_side(side_tex, is_training=True)

                    loss_fake_back = self.calculate_d_loss(
                        d_back, mask_d_back, z_style_idx, is_real=False
                    )
                    loss_fake_front = self.calculate_d_loss(
                        d_front, mask_d_front, z_style_idx, is_real=False
                    )
                    loss_fake_top = self.calculate_d_loss(
                        d_top, mask_d_top, z_style_idx, is_real=False
                    )
                    loss_fake_side = self.calculate_d_loss(
                        d_side, mask_d_side, z_style_idx, is_real=False
                    )

                    total_gen = (
                        loss_fake_back
                        + loss_fake_front
                        + loss_fake_top
                        + loss_fake_side
                    )
                    total_gen.backward()
                    self.optim_d_tex.step()

                ### Reconstruction training
                for _ in range(self.r_steps):

                    self.generator.zero_grad()
                    z_torch_geo_r, z_torch_tex_r, z_style_idx_r = (
                        self.sample_style_tex()
                    )
                    geo_code_r = torch.matmul(
                        z_torch_geo_r, self.generator.geometry_codes
                    ).view([1, -1, 1, 1, 1])
                    tex_code_r = torch.matmul(
                        z_torch_tex_r, self.generator.texture_codes
                    ).view([1, -1, 1, 1, 1])

                    style_batch_r = self.style_dataset.__getitem__(z_style_idx_r)
                    renders_list = style_batch_r["renders"].to(self.device)
                    (
                        style_back_tex_r,
                        style_front_tex_r,
                        style_top_tex_r,
                        style_side_tex_r,
                    ) = (
                        renders_list[0].permute(2, 0, 1).unsqueeze(0).float(),
                        renders_list[1].permute(2, 0, 1).unsqueeze(0).float(),
                        renders_list[2].permute(2, 0, 1).unsqueeze(0).float(),
                        renders_list[3].permute(2, 0, 1).unsqueeze(0).float(),
                    )
                    style_geo_in = (
                        style_batch_r["geo_in"].to(self.device).unsqueeze(0).float()
                    )
                    style_mask_g = (
                        style_batch_r["mask_g"].to(self.device).unsqueeze(0).float()
                    )

                    geo_gen_r, tex_gen_r = self.generator(
                        voxels=style_geo_in,
                        geometry_z=geo_code_r,
                        texture_z=tex_code_r,
                        mask=style_mask_g,
                        is_geometry_training=False,
                    )

                    back_tex_r, front_tex_r, top_tex_r, side_tex_r = render_voxel(
                        (geo_gen_r > self.threshold).float(), tex_gen_r
                    )

                    total_r_loss = (
                        self.calculate_r_tex_loss(
                            gen=back_tex_r, gt=style_back_tex_r, beta=self.beta
                        )
                        + self.calculate_r_tex_loss(
                            gen=front_tex_r, gt=style_front_tex_r, beta=self.beta
                        )
                        + self.calculate_r_tex_loss(
                            gen=top_tex_r, gt=style_top_tex_r, beta=self.beta
                        )
                        + self.calculate_r_tex_loss(
                            gen=side_tex_r, gt=style_side_tex_r, beta=self.beta
                        )
                    )
                    total_r_loss.backward()

                    self.optim_g.step()

                ### Generator training
                for _ in range(self.g_steps):

                    geo_code = torch.matmul(
                        z_torch_geo, self.generator.geometry_codes
                    ).view([1, -1, 1, 1, 1])

                    tex_code = torch.matmul(
                        z_torch_tex, self.generator.texture_codes
                    ).view([1, -1, 1, 1, 1])

                    geo_gen, tex_gen = self.generator(
                        voxels=cnt_geo_in,
                        geometry_z=geo_code,
                        texture_z=tex_code,
                        mask=cnt_mask_g,
                        is_geometry_training=False,
                    )

                    back_tex, front_tex, top_tex, side_tex = render_voxel(
                        (geo_gen > self.threshold).float(), tex_gen
                    )

                    d_back = self.tex_d_back(back_tex, is_training=False)
                    d_front = self.tex_d_front(front_tex, is_training=False)
                    d_top = self.tex_d_top(top_tex, is_training=False)
                    d_side = self.tex_d_side(side_tex, is_training=False)

                    loss_g = (
                        self.calculate_g_loss(
                            d_back, mask_d_back, z_style_idx, self.alpha
                        )
                        + self.calculate_g_loss(
                            d_front, mask_d_front, z_style_idx, self.alpha
                        )
                        + self.calculate_g_loss(
                            d_top, mask_d_top, z_style_idx, self.alpha
                        )
                        + self.calculate_g_loss(
                            d_side, mask_d_side, z_style_idx, self.alpha
                        )
                    )
                    loss_g.backward()
                    self.optim_g.step()

            report_str = f"Epoch: {epoch}, loss_g: {loss_g.item()}, loss_r: {total_r_loss.item()}, loss_d: {total_gen.item()}"
            print(report_str)

    def calculate_r_tex_loss(self, gen, gt, beta):
        total = torch.mean(torch.abs(gen - gt) ** 2) * beta
        return total

    def calculate_g_loss(self, d, mask, z_idx, alpha):

        diff_local = (d[:, z_idx : z_idx + 1] - 1) ** 2
        diff_global = (d[:, -1:] - 1) ** 2

        masked_local = torch.sum(diff_local * mask) * alpha
        masked_global = torch.sum(diff_global * mask)
        total = (masked_local + masked_global) / torch.sum(mask)
        return total

    def calculate_r_loss(self, gen_s, gen_l, gt_s, gt_l, beta):
        loss_s = torch.mean(torch.abs(gen_s - gt_s) ** 2) * beta
        loss_l = torch.mean(torch.abs(gen_l - gt_l) ** 2) * beta
        total = loss_s + loss_l
        return total

    def calculate_d_loss(self, d, mask, z_idx, is_real):

        if is_real:
            diff_local = (d[:, z_idx : z_idx + 1] - 1) ** 2
            diff_global = (d[:, -1:] - 1) ** 2
        else:
            diff_local = (d[:, z_idx : z_idx + 1]) ** 2
            diff_global = (d[:, -1:]) ** 2

        masked_local = torch.sum(diff_local * mask)
        masked_global = torch.sum(diff_global * mask)
        total = (masked_local + masked_global) / torch.sum(mask)
        return total

    def sample_random_style(self):
        z_ = np.zeros([self.n_style], np.float32)
        z_idx = np.random.randint(self.n_style)
        z_[z_idx] = 1
        z_torch = torch.from_numpy(z_).to(self.device).view([1, -1])
        return z_torch, z_idx

    def sample_style_tex(self):
        idx = np.random.randint(self.n_style)
        z_geo = np.zeros([self.n_style], np.float32)
        z_tex = np.zeros([self.n_style], np.float32)
        z_geo[idx] = 1
        z_tex[idx] = 1
        z_geo_torch = torch.from_numpy(z_geo).to(self.device).view([1, -1])
        z_tex_torch = torch.from_numpy(z_tex).to(self.device).view([1, -1])
        return z_geo_torch, z_tex_torch, idx

    def compute_loss(self, output):
        pass

    def post_epoch_actions(self, epoch):
        pass
