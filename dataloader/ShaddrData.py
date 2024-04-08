import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


class ShaddrDataset(Dataset):
    def __init__(self, root_path, is_style, split_file):
        self.root_path = root_path
        self.input_size = 32
        self.output_size = 256
        self.upsample_rate = 8
        self.mask_margin = 16
        self.asymmetry = False
        self.is_style = is_style
        self.split_file = split_file
        self.device = torch.device("cuda")

        self.data_paths = self.make_path_list(self.split_file)

    def __len__(self):
        return len(self.data_paths)

    def make_path_list(self, split_file):
        with open(split_file) as fhand:
            path_list = [os.path.join(self.root_path, line.rstrip()) for line in fhand]
        return path_list

    def read_voxel_file(self, filename, fix_coords=True):
        fhand = open(filename, "rb")

        # Read Header
        line = fhand.readline().strip()
        if not line.startswith(b"#binvox"):
            raise IOError("Not a binvox file")

        dims = [int(i) for i in fhand.readline().strip().split(b" ")[1:]]
        fhand.readline()  # omit translate
        fhand.readline()  # omit scale
        fhand.readline()  # omit "data\n"

        raw_data = np.frombuffer(fhand.read(), dtype=np.uint8)
        fhand.close()

        values, counts = raw_data[::2], raw_data[1::2]
        data = np.repeat(values, counts).astype(np.bool_)
        data = data.reshape(dims).astype(np.uint8)
        if fix_coords:
            data = np.ascontiguousarray(np.transpose(data, (0, 2, 1)))
        return data

    def read_hdf5_file(self, filename):
        data_dict = h5py.File(filename, "r")
        voxel_texture = data_dict["voxel_color"][:]
        data_dict.close()

        geometry = voxel_texture[:, :, :, -1]
        texture = voxel_texture[:, :, :, :3]
        texture = texture[:, :, :, [2, 1, 0]]

        return torch.cat([texture, geometry], dim=3)

    def get_voxel_bbox(self, vox):
        # minimap
        vox_tensor = (
            torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        )
        smallmaskx_tensor = F.max_pool3d(
            vox_tensor,
            kernel_size=self.upsample_rate,
            stride=self.upsample_rate,
            padding=0,
        )
        smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0, 0]
        smallmaskx = np.round(smallmaskx).astype(np.uint8)
        smallx, smally, smallz = smallmaskx.shape
        # x
        ray = np.max(smallmaskx, (1, 2))
        xmin = 0
        xmax = 0
        for i in range(smallx):
            if ray[i] > 0:
                if xmin == 0:
                    xmin = i
                xmax = i
        # y
        ray = np.max(smallmaskx, (0, 2))
        ymin = 0
        ymax = 0
        for i in range(smally):
            if ray[i] > 0:
                if ymin == 0:
                    ymin = i
                ymax = i
        # z
        ray = np.max(smallmaskx, (0, 1))
        if self.asymmetry:
            zmin = 0
            zmax = 0
            for i in range(smallz):
                if ray[i] > 0:
                    if zmin == 0:
                        zmin = i
                    zmax = i
        else:
            zmin = smallz // 2
            zmax = 0
            for i in range(zmin, smallz):
                if ray[i] > 0:
                    zmax = i

        return xmin, xmax + 1, ymin, ymax + 1, zmin, zmax + 1

    def crop_voxel(self, vox, xmin, xmax, ymin, ymax, zmin, zmax):
        xspan = xmax - xmin
        yspan = ymax - ymin
        zspan = zmax - zmin
        tmp = np.zeros(
            [
                xspan * self.upsample_rate + self.mask_margin * 2,
                yspan * self.upsample_rate + self.mask_margin * 2,
                zspan * self.upsample_rate + self.mask_margin * 2,
            ],
            np.uint8,
        )
        if self.asymmetry:
            tmp[
                self.mask_margin : -self.mask_margin,
                self.mask_margin : -self.mask_margin,
                self.mask_margin : -self.mask_margin,
            ] = vox[
                xmin * self.upsample_rate : xmax * self.upsample_rate,
                ymin * self.upsample_rate : ymax * self.upsample_rate,
                zmin * self.upsample_rate : zmax * self.upsample_rate,
            ]
        else:
            # note z is special: only get half of the shape in z:  0     0.5-----1
            tmp[
                self.mask_margin : -self.mask_margin,
                self.mask_margin : -self.mask_margin,
                : -self.mask_margin,
            ] = vox[
                xmin * self.upsample_rate : xmax * self.upsample_rate,
                ymin * self.upsample_rate : ymax * self.upsample_rate,
                zmin * self.upsample_rate
                - self.mask_margin : zmax * self.upsample_rate,
            ]
        return tmp

    def crop_color_voxel(self, color_vox, xmin, xmax, ymin, ymax, zmin, zmax):
        xspan = xmax - xmin
        yspan = ymax - ymin
        zspan = zmax - zmin
        tmp = np.zeros(
            [
                xspan * self.upsample_rate + self.mask_margin * 2,
                yspan * self.upsample_rate + self.mask_margin * 2,
                zspan * self.upsample_rate + self.mask_margin * 2,
                3,
            ],
            np.float32,
        )
        if self.asymmetry:
            tmp[
                self.mask_margin : -self.mask_margin,
                self.mask_margin : -self.mask_margin,
                self.mask_margin : -self.mask_margin,
                :,
            ] = color_vox[
                xmin * self.upsample_rate : xmax * self.upsample_rate,
                ymin * self.upsample_rate : ymax * self.upsample_rate,
                zmin * self.upsample_rate : zmax * self.upsample_rate,
                :,
            ]
        else:
            # note z is special: only get half of the shape in z:  0     0.5-----1
            tmp[
                self.mask_margin : -self.mask_margin,
                self.mask_margin : -self.mask_margin,
                : -self.mask_margin,
                :,
            ] = color_vox[
                xmin * self.upsample_rate : xmax * self.upsample_rate,
                ymin * self.upsample_rate : ymax * self.upsample_rate,
                zmin * self.upsample_rate
                - self.mask_margin : zmax * self.upsample_rate,
                :,
            ]

        return tmp

    def get_style_voxel_Dmask(self, vox):
        if (
            self.upsample_rate == 8
            and self.input_size == 32
            and self.output_size == 256
        ):
            # 256 -maxpoolk8s8- 32 -crop- 30 -upsample- 120
            # output: 56, 120
            crop_margin_1 = 1
            crop_margin_2 = 2
            scale_factor_1 = 2
            scale_factor_2 = 4
            upsample_rate = self.upsample_rate

        # Dmask contains the whole voxel (surface + inside)
        vox_tensor = (
            torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        )

        smallmaskx_tensor = F.max_pool3d(
            vox_tensor, kernel_size=upsample_rate, stride=upsample_rate, padding=0
        )
        smallmask_tensor_rfl = smallmaskx_tensor[
            :,
            :,
            crop_margin_1:-crop_margin_1,
            crop_margin_1:-crop_margin_1,
            crop_margin_1:-crop_margin_1,
        ]
        if self.input_size == 32 and self.output_size == 256:
            smallmask_tensor_rfl = F.max_pool3d(
                smallmask_tensor_rfl, kernel_size=3, stride=1, padding=1
            )
        smallmask_tensor_rfl = F.interpolate(
            smallmask_tensor_rfl,
            scale_factor=upsample_rate // scale_factor_1,
            mode="nearest",
        )

        smallmask_tensor_rfs = smallmaskx_tensor[
            :,
            :,
            crop_margin_2:-crop_margin_2,
            crop_margin_2:-crop_margin_2,
            crop_margin_2:-crop_margin_2,
        ]
        smallmask_tensor_rfs = F.interpolate(
            smallmask_tensor_rfs,
            scale_factor=upsample_rate // scale_factor_2,
            mode="nearest",
        )

        smallmask_rfl = smallmask_tensor_rfl.detach().cpu().numpy()[0, 0]
        smallmask_rfs = smallmask_tensor_rfs.detach().cpu().numpy()[0, 0]
        smallmask_rfl = np.round(smallmask_rfl).astype(np.uint8)
        smallmask_rfs = np.round(smallmask_rfs).astype(np.uint8)

        return smallmask_rfl, smallmask_rfs

    def get_voxel_input_Dmask_Gmask(self, vox):
        if (
            self.upsample_rate == 8
            and self.input_size == 32
            and self.output_size == 256
        ):
            # 512 -maxpoolk8s8- 64 -crop- 60 -upsample- 120
            # output: 56, 120
            crop_margin_1 = 1
            crop_margin_2 = 2
            scale_factor_1 = 2
            scale_factor_2 = 4
            upsample_rate = self.upsample_rate

        vox_tensor = (
            torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        )
        # input
        smallmaskx_tensor = F.max_pool3d(
            vox_tensor, kernel_size=upsample_rate, stride=upsample_rate, padding=0
        )

        # Dmask contains the whole voxel (surface + inside)
        smallmask_tensor_rfl = smallmaskx_tensor[
            :,
            :,
            crop_margin_1:-crop_margin_1,
            crop_margin_1:-crop_margin_1,
            crop_margin_1:-crop_margin_1,
        ]
        if self.input_size == 32 and self.output_size == 256:
            smallmask_tensor_rfl = F.max_pool3d(
                smallmask_tensor_rfl, kernel_size=3, stride=1, padding=1
            )
        smallmask_tensor_rfl = F.interpolate(
            smallmask_tensor_rfl,
            scale_factor=upsample_rate // scale_factor_1,
            mode="nearest",
        )

        smallmask_tensor_rfs = smallmaskx_tensor[
            :,
            :,
            crop_margin_2:-crop_margin_2,
            crop_margin_2:-crop_margin_2,
            crop_margin_2:-crop_margin_2,
        ]
        smallmask_tensor_rfs = F.interpolate(
            smallmask_tensor_rfs,
            scale_factor=upsample_rate // scale_factor_2,
            mode="nearest",
        )

        # Gmask
        # expand 1
        if self.upsample_rate == 8:
            mask_tensor = smallmaskx_tensor

        mask_tensor = F.max_pool3d(mask_tensor, kernel_size=3, stride=1, padding=1)

        # to numpy
        smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0, 0]
        smallmask_rfl = smallmask_tensor_rfl.detach().cpu().numpy()[0, 0]
        smallmask_rfs = smallmask_tensor_rfs.detach().cpu().numpy()[0, 0]
        mask = mask_tensor.detach().cpu().numpy()[0, 0]
        smallmaskx = np.round(smallmaskx).astype(np.uint8)
        smallmask_rfl = np.round(smallmask_rfl).astype(np.uint8)
        smallmask_rfs = np.round(smallmask_rfs).astype(np.uint8)
        mask = np.round(mask).astype(np.uint8)

        return smallmaskx, smallmask_rfl, smallmask_rfs, mask

    def get_rendered_views(self, color_path, xmin, xmax, ymin, ymax, zmin, zmax):
        data_dict = h5py.File(color_path, "r")
        voxel_texture = data_dict["voxel_color"][:]
        data_dict.close()

        # the color is BGR, change to RGB
        geometry = voxel_texture[:, :, :, -1]
        texture = voxel_texture[:, :, :, :3]
        texture = texture[:, :, :, [2, 1, 0]]

        if self.output_size == 256:
            geometry = F.max_pool3d(
                torch.from_numpy(geometry).unsqueeze(0).unsqueeze(0).float(),
                kernel_size=2,
                stride=2,
                padding=0,
            ).numpy()[0, 0]
            texture = (
                F.interpolate(
                    torch.from_numpy(texture).permute(3, 0, 1, 2).unsqueeze(0).float(),
                    scale_factor=0.5,
                    mode="trilinear",
                )
                .squeeze(0)
                .permute(1, 2, 3, 0)
                .numpy()
            )
            assert (texture >= 0).all()

        # crop voxel color same as geometry
        geometry_crop = self.crop_voxel(geometry, xmin, xmax, ymin, ymax, zmin, zmax)
        texture_crop = self.crop_color_voxel(
            texture, xmin, xmax, ymin, ymax, zmin, zmax
        )
        geometry_crop = (
            torch.from_numpy(geometry_crop)
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )
        texture_crop = (
            torch.from_numpy(texture_crop)
            .to(self.device)
            .permute(3, 0, 1, 2)
            .contiguous()
            .unsqueeze(0)
            .float()
            / 255.0
        )

        back_texture, front_texture, top_texture, left_texture, right_texture = (
            self.rendering(geometry_crop, texture_crop)
        )

        # each (512, 512, 3+1)
        back_texture = (
            back_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
        )
        front_texture = (
            front_texture.squeeze(0)
            .permute(1, 2, 0)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )
        top_texture = (
            top_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
        )
        left_texture = (
            left_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
        )
        right_texture = (
            right_texture.squeeze(0)
            .permute(1, 2, 0)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        return back_texture, front_texture, top_texture, left_texture, right_texture

    def rendering(self, geometry_tensor, texture_tensor):
        # back
        _, _, dim_x, dim_y, dim_z = geometry_tensor.size()
        back_mask, back_depth = torch.max(geometry_tensor[0, 0], 0)
        texture = torch.cat(
            [
                torch.gather(texture_tensor[0, 0], 0, back_depth.unsqueeze(0)),
                torch.gather(texture_tensor[0, 1], 0, back_depth.unsqueeze(0)),
                torch.gather(texture_tensor[0, 2], 0, back_depth.unsqueeze(0)),
            ],
            0,
        )
        back_texture = texture * back_mask.unsqueeze(0)  # (3, 512, 512)

        # front
        front_mask, front_depth = torch.max(geometry_tensor[0, 0].flip(0), 0)
        texture = torch.cat(
            [
                torch.gather(
                    texture_tensor[0, 0], 0, dim_x - 1 - front_depth.unsqueeze(0)
                ),
                torch.gather(
                    texture_tensor[0, 1], 0, dim_x - 1 - front_depth.unsqueeze(0)
                ),
                torch.gather(
                    texture_tensor[0, 2], 0, dim_x - 1 - front_depth.unsqueeze(0)
                ),
            ],
            0,
        )
        front_texture = texture * front_mask.unsqueeze(0)  # (3, 512, 512)

        # top
        top_mask, top_depth = torch.max(geometry_tensor[0, 0].flip(1), 1)
        texture = torch.cat(
            [
                torch.gather(
                    texture_tensor[0, 0], 1, dim_y - 1 - top_depth.unsqueeze(1)
                ),
                torch.gather(
                    texture_tensor[0, 1], 1, dim_y - 1 - top_depth.unsqueeze(1)
                ),
                torch.gather(
                    texture_tensor[0, 2], 1, dim_y - 1 - top_depth.unsqueeze(1)
                ),
            ],
            1,
        )
        top_texture = texture.permute(1, 0, 2) * top_mask.unsqueeze(
            0
        )  # (512, 3, 512) -> (3, 512, 512)

        # side - left
        left_mask, left_depth = torch.max(geometry_tensor[0, 0].flip(2), 2)
        texture = torch.cat(
            [
                torch.gather(
                    texture_tensor[0, 0], 2, dim_z - 1 - left_depth.unsqueeze(2)
                ),
                torch.gather(
                    texture_tensor[0, 1], 2, dim_z - 1 - left_depth.unsqueeze(2)
                ),
                torch.gather(
                    texture_tensor[0, 2], 2, dim_z - 1 - left_depth.unsqueeze(2)
                ),
            ],
            2,
        )
        left_texture = texture.permute(2, 0, 1) * left_mask.unsqueeze(
            0
        )  # (512, 512, 3) -> (3, 512, 512)
        # left_mask = self.fill_geometry_mask(left_mask, fill_x=True)

        # side - right, [only needed when asymmetry]
        right_mask, right_depth = torch.max(geometry_tensor[0, 0], 2)
        texture = torch.cat(
            [
                torch.gather(texture_tensor[0, 0], 2, right_depth.unsqueeze(2)),
                torch.gather(texture_tensor[0, 1], 2, right_depth.unsqueeze(2)),
                torch.gather(texture_tensor[0, 2], 2, right_depth.unsqueeze(2)),
            ],
            2,
        )
        right_texture = texture.permute(2, 0, 1) * right_mask.unsqueeze(
            0
        )  # (512, 512, 3) -> (3, 512, 512)

        # each (1, 3+1, 512, 512)
        back_texture = torch.cat(
            (back_texture, back_mask.unsqueeze(0)), dim=0
        ).unsqueeze(0)
        front_texture = torch.cat(
            (front_texture, front_mask.unsqueeze(0)), dim=0
        ).unsqueeze(0)
        top_texture = torch.cat((top_texture, top_mask.unsqueeze(0)), dim=0).unsqueeze(
            0
        )
        left_texture = torch.cat(
            (left_texture, left_mask.unsqueeze(0)), dim=0
        ).unsqueeze(0)
        right_texture = torch.cat(
            (right_texture, right_mask.unsqueeze(0)), dim=0
        ).unsqueeze(0)

        return back_texture, front_texture, top_texture, left_texture, right_texture

    def __getitem__(self, idx):
        path = self.data_paths[idx]
        binvox_file = os.path.join(path, "model_depth_fusion.binvox")
        hdf5_file = os.path.join(path, "voxel_color.hdf5")

        voxel_data = self.read_voxel_file(binvox_file)

        downsample_size = int(512 / self.output_size)
        voxel_data = (
            F.max_pool3d(
                torch.tensor(voxel_data, dtype=torch.float32).unsqueeze(0),
                downsample_size,
                downsample_size,
            )
            .squeeze(0)
            .numpy()
            .astype(np.uint8)
        )

        xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(voxel_data)
        voxels = self.crop_voxel(voxel_data, xmin, xmax, ymin, ymax, zmin, zmax)

        if self.is_style == True:
            voxel_style_lg = gaussian_filter(voxels.astype(np.float32), sigma=1)
            voxels_sm = F.max_pool3d(
                torch.from_numpy(voxels.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                kernel_size=2,
                stride=2,
                padding=0,
            ).numpy()[0, 0]
            voxel_style_sm = gaussian_filter(voxels_sm.astype(np.float32), sigma=1)
            dmask_style_lg, dmask_style_sm = self.get_style_voxel_Dmask(voxels)

            input_style, _, _, gmask_style = self.get_voxel_input_Dmask_Gmask(voxels)
            pos_style = [xmin, xmax, ymin, ymax, zmin, zmax]

            back, front, top, left, right = self.get_rendered_views(
                hdf5_file, xmin, xmax, ymin, ymax, zmin, zmax
            )

            net_input = {
                "mask_g": torch.from_numpy(gmask_style).unsqueeze(0),
                "mask_d_l": torch.from_numpy(dmask_style_lg).unsqueeze(0),
                "mask_d_s": torch.from_numpy(dmask_style_sm).unsqueeze(0),
                "geo_l": torch.from_numpy(voxel_style_lg).unsqueeze(0),
                "geo_s": torch.from_numpy(voxel_style_sm).unsqueeze(0),
                "geo_in": torch.from_numpy(input_style).unsqueeze(0),
                "bbox": torch.tensor(pos_style).unsqueeze(0),
                "back_render": torch.from_numpy(back).unsqueeze(0),
                "front_render": torch.from_numpy(front).unsqueeze(0),
                "top_render": torch.from_numpy(top).unsqueeze(0),
                "side_render": torch.from_numpy(left).unsqueeze(0),
            }

            return net_input
        else:
            voxel_content_lg = gaussian_filter(voxels.astype(np.float32), sigma=1)
            voxels_sm = F.max_pool3d(
                torch.from_numpy(voxels.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                kernel_size=2,
                stride=2,
                padding=0,
            ).numpy()[0, 0]
            voxel_content_sm = gaussian_filter(voxels_sm.astype(np.float32), sigma=1)
            input_content, dmask_content_lg, dmask_content_sm, gmask_content = (
                self.get_voxel_input_Dmask_Gmask(voxels)
            )
            pos_content = [xmin, xmax, ymin, ymax, zmin, zmax]

            net_input = {
                "mask_g": torch.from_numpy(gmask_content),
                "mask_d_l": torch.from_numpy(dmask_content_lg),
                "mask_d_s": torch.from_numpy(dmask_content_sm),
                "geo_l": torch.from_numpy(voxel_content_lg),
                "geo_s": torch.from_numpy(voxel_content_sm),
                "geo_in": torch.from_numpy(input_content),
                "bbox": torch.tensor(pos_content),
            }

            return net_input


# if __name__ == "__main__":
#     root_path = r"C:\Users\chara\OneDrive\Desktop\Frontiers of VC\Project\data\Shapenet_Voxels\03001627"
#     is_style = True
#     split_file = r"C:\Users\chara\OneDrive\Desktop\Frontiers of VC\Project\temp_project_folder\splits\style_color_chair_4.txt"

#     style_dataset = ShaddrDataset(root_path, is_style, split_file)

#     root_path = r"C:\Users\chara\OneDrive\Desktop\Frontiers of VC\Project\data\Shapenet_Voxels\03001627"
#     is_style = False
#     split_file = r"C:\Users\chara\OneDrive\Desktop\Frontiers of VC\Project\temp_project_folder\splits\content_chair_train.txt"

#     content_dataset = ShaddrDataset(root_path, is_style, split_file)

#     from torch.utils.data import DataLoader

#     content_dataloader = DataLoader(content_dataset, batch_size=1)
#     style_dataloader = DataLoader(style_dataset, batch_size=1)

#     # for i, net_input in enumerate(content_dataloader):
#     #     print(i, net_input)
#     for i, net_input in enumerate(style_dataloader):
#         print(i, net_input)
