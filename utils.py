### Contains utility functions for the project. some of the functions are taken from different parts of https://github.com/qiminchen/ShaDDR/tree/main with some modifications.

import torch
import torch.nn.functional as F
import numpy as np


def recover_voxel(vox, bbox):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]
    umsample_rate = 8
    real_size = 256
    mask_margin = 16
    tmpvox = np.zeros([real_size, real_size, real_size], np.float32)
    xmin_, ymin_, zmin_ = (0, 0, 0)
    xmax_, ymax_, zmax_ = vox.shape
    xmin = xmin * umsample_rate - mask_margin
    xmax = xmax * umsample_rate + mask_margin
    ymin = ymin * umsample_rate - mask_margin
    ymax = ymax * umsample_rate + mask_margin

    zmin = zmin * umsample_rate
    zmin_ = mask_margin

    zmax = zmax * umsample_rate + mask_margin

    if xmin < 0:
        xmin_ = -xmin
        xmin = 0
    if xmax > real_size:
        xmax_ = xmax_ + real_size - xmax
        xmax = real_size
    if ymin < 0:
        ymin_ = -ymin
        ymin = 0
    if ymax > real_size:
        ymax_ = ymax_ + real_size - ymax
        ymax = real_size
    if zmin < 0:
        zmin_ = -zmin
        zmin = 0
    if zmax > real_size:
        zmax_ = zmax_ + real_size - zmax
        zmax = real_size

    tmpvox[xmin:xmax, ymin:ymax, zmin:zmax] = vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_]
    if zmin * 2 - zmax - 1 < 0:
        tmpvox[xmin:xmax, ymin:ymax, zmin - 1 :: -1] = vox[
            xmin_:xmax_, ymin_:ymax_, zmin_:zmax_
        ]
    else:
        tmpvox[xmin:xmax, ymin:ymax, zmin - 1 : zmin * 2 - zmax - 1 : -1] = vox[
            xmin_:xmax_, ymin_:ymax_, zmin_:zmax_
        ]

    return tmpvox


def render_voxel(geometry_tensor, texture_tensor):
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
            torch.gather(texture_tensor[0, 0], 0, dim_x - 1 - front_depth.unsqueeze(0)),
            torch.gather(texture_tensor[0, 1], 0, dim_x - 1 - front_depth.unsqueeze(0)),
            torch.gather(texture_tensor[0, 2], 0, dim_x - 1 - front_depth.unsqueeze(0)),
        ],
        0,
    )
    front_texture = texture * front_mask.unsqueeze(0)  # (3, 512, 512)

    # top
    top_mask, top_depth = torch.max(geometry_tensor[0, 0].flip(1), 1)
    texture = torch.cat(
        [
            torch.gather(texture_tensor[0, 0], 1, dim_y - 1 - top_depth.unsqueeze(1)),
            torch.gather(texture_tensor[0, 1], 1, dim_y - 1 - top_depth.unsqueeze(1)),
            torch.gather(texture_tensor[0, 2], 1, dim_y - 1 - top_depth.unsqueeze(1)),
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
            torch.gather(texture_tensor[0, 0], 2, dim_z - 1 - left_depth.unsqueeze(2)),
            torch.gather(texture_tensor[0, 1], 2, dim_z - 1 - left_depth.unsqueeze(2)),
            torch.gather(texture_tensor[0, 2], 2, dim_z - 1 - left_depth.unsqueeze(2)),
        ],
        2,
    )
    left_texture = texture.permute(2, 0, 1) * left_mask.unsqueeze(
        0
    )  # (512, 512, 3) -> (3, 512, 512)
    # left_mask = self.fill_geometry_mask(left_mask, fill_x=True)

    # each (1, 3+1, 512, 512)
    back_texture = torch.cat((back_texture, back_mask.unsqueeze(0)), dim=0).unsqueeze(0)
    front_texture = torch.cat(
        (front_texture, front_mask.unsqueeze(0)), dim=0
    ).unsqueeze(0)
    top_texture = torch.cat((top_texture, top_mask.unsqueeze(0)), dim=0).unsqueeze(0)
    left_texture = torch.cat((left_texture, left_mask.unsqueeze(0)), dim=0).unsqueeze(0)

    return back_texture, front_texture, top_texture, left_texture


def get_view_mask_d(view):
    mask = F.max_pool2d(view, kernel_size=8, stride=8, padding=0)
    mask = mask[:, :, 1:-1, 1:-1]
    mask = F.interpolate(mask, scale_factor=4, mode="nearest")

    return mask


def get_tex_mask_d(back, front, top, side):
    masks = []
    for view in [back, front, top, side]:
        v = view[:, 3:, :, :]
        mask = get_view_mask_d(v)
        masks.append(mask)

    return tuple(masks)
