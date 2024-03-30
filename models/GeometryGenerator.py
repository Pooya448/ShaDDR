import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryGenerator(nn.Module):
    """
    A generator module designed for geometry synthesis, implementing a dual-branch architecture
    to generate geometry at two different scales (half-size x8 small).

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

        # Define the convolutional layers for geometry synthesis
        self.conv_5 = nn.ConvTranspose3d(
            dim_g * 8 + dim_z, dim_g * 4, 4, stride=2, padding=1, bias=True
        )
        self.conv_6 = nn.Conv3d(
            dim_g * 4 + dim_z, dim_g * 4, 3, stride=1, padding=1, bias=True
        )

        self.conv_7 = nn.ConvTranspose3d(
            dim_g * 4 + dim_z, dim_g * 2, 4, stride=2, padding=1, bias=True
        )
        self.conv_8 = nn.Conv3d(
            dim_g * 2 + dim_z, dim_g * 2, 3, stride=1, padding=1, bias=True
        )
        self.conv_8_out = nn.Conv3d(dim_g * 2, 1, 3, stride=1, padding=1, bias=True)

        self.conv_9 = nn.ConvTranspose3d(
            dim_g * 2 + dim_z, dim_g, 4, stride=2, padding=1, bias=True
        )
        self.conv_10 = nn.Conv3d(
            dim_g + dim_z, dim_g, 3, stride=1, padding=1, bias=True
        )
        self.conv_10_out = nn.Conv3d(dim_g, 1, 3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, mask_, is_training=False):
        """
        Forward pass through the generator.

        Processes input data alongside latent space representations and masks through a series of
        convolutional layers to generate geometry at two scales.

        Parameters:
            voxels (torch.Tensor): Input voxel data.
            z (torch.Tensor): Latent space representation to be integrated at each stage.
            mask_ (torch.Tensor): Input mask for geometry synthesis.
            is_training (bool, optional): Flag indicating if the model is in training mode. Defaults to False.

        Returns:
            tuple: A tuple containing two torch.Tensors representing the generated geometry at two scales.
        """
        out = voxels
        mask_256 = F.interpolate(mask_, scale_factor=8, mode="nearest")
        mask_128 = F.interpolate(mask_, scale_factor=4, mode="nearest")

        # Sequential processing and integration of latent representation at each stage
        for layer in [
            self.conv_5,
            self.conv_6,
            self.conv_7,
            self.conv_8,
            self.conv_9,
            self.conv_10,
        ]:
            _, _, dimx, dimy, dimz = out.size()
            zs = z.repeat(1, 1, dimx, dimy, dimz)
            out = torch.cat([out, zs], dim=1)
            out = F.leaky_relu(layer(out), negative_slope=0.02, inplace=True)
            if layer in [self.conv_8, self.conv_10]:
                out = layer(out)
                out = torch.max(torch.min(out, out * 0.002 + 0.998), out * 0.002)
                if layer == self.conv_8:
                    out_128 = out * mask_128
                elif layer == self.conv_10:
                    out_256 = out * mask_256

        return out_256, out_128
