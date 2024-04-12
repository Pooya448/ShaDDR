import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryGenerator(nn.Module):
    """
    A generator module designed for geometry synthesis, implementing a dual-branch architecture
    to generate geometry at two different scales, 32 and 256.

    This model progressively decodes the feature representation into spatial dimensions,
    using transposed convolutions to upscale and regular convolutions to refine the features
    at each scale. It integrates latent space representations at various stages of the pipeline.

    Attributes:
        dim_g (int): Base dimensionality of the generator's convolutional layers.
        dim_z (int): Dimensionality of the latent space representation.
    """

    def __init__(self, dim_g, dim_z):
        """
        Initializes the GeometryGenerator model with specified dimensions.

        Parameters:
            dim_g (int): Base dimensionality for the generator's convolutional layers.
            dim_z (int): Dimensionality of the latent space representation.
        """
        super(GeometryGenerator, self).__init__()
        self.dim_g = dim_g
        self.dim_z = dim_z

        self.layers = nn.ModuleList(
            [
                nn.ConvTranspose3d(
                    dim_g * 8 + dim_z, dim_g * 4, 4, stride=2, padding=1, bias=True
                ),
                nn.Conv3d(
                    dim_g * 4 + dim_z, dim_g * 4, 3, stride=1, padding=1, bias=True
                ),
                nn.ConvTranspose3d(
                    dim_g * 4 + dim_z, dim_g * 2, 4, stride=2, padding=1, bias=True
                ),
                nn.Conv3d(
                    dim_g * 2 + dim_z, dim_g * 2, 3, stride=1, padding=1, bias=True
                ),
                nn.ConvTranspose3d(
                    dim_g * 2 + dim_z, dim_g, 4, stride=2, padding=1, bias=True
                ),
                nn.Conv3d(dim_g + dim_z, dim_g, 3, stride=1, padding=1, bias=True),
            ]
        )

        self.s_conv = nn.Conv3d(dim_g * 2, 1, 3, stride=1, padding=1, bias=True)

        self.l_conv = nn.Conv3d(dim_g, 1, 3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, mask):
        """
        Forward pass through the generator.

        Processes input data alongside latent space representations and masks through a series of
        convolutional layers to generate geometry at two scales.

        Parameters:
            voxels (torch.Tensor): Input voxel data.
            z (torch.Tensor): Latent space representation to be integrated at each stage.
            mask (torch.Tensor): Input mask for geometry synthesis.

        Returns:
            tuple: A tuple containing two torch.Tensors representing the generated geometry at two scales.
        """
        out = voxels
        mask_l = F.interpolate(mask, scale_factor=8, mode="nearest")
        mask_s = F.interpolate(mask, scale_factor=4, mode="nearest")

        for i, layer in enumerate(self.layers):
            _, _, dimx, dimy, dimz = out.size()
            zs = z.repeat(1, 1, dimx, dimy, dimz)
            out = torch.cat([out, zs], dim=1)
            out = layer(out)
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

            if i == 3:
                out_s = self.s_conv(out)
                out_s = torch.max(
                    torch.min(out_s, out_s * 0.002 + 0.998), out_s * 0.002
                )
            elif i == 5:
                out_l = self.l_conv(out)
                out_l = torch.max(
                    torch.min(out_l, out_l * 0.002 + 0.998), out_l * 0.002
                )

        out_l = out_l * mask_l
        out_s = out_s * mask_s

        return out_l, out_s
