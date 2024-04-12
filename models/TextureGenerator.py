import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureGenerator(nn.Module):
    """
    A generator module designed for texture synthesis, expanding the input dimensionality
    by a factor of 8 with a dual-branch small architecture.

    This model decodes the feature representation into texture details at multiple scales,
    using a series of transposed and regular 3D convolutions. It integrates a latent space
    representation at various stages to enrich the generated textures.

    Attributes:
        dim_g (int): Base dimensionality of the generator's convolutional layers.
        dim_z (int): Dimensionality of the latent space representation.
    """

    def __init__(self, dim_g, dim_z):
        """
        Initializes the TextureGenerator model with specified dimensions.

        Parameters:
            dim_g (int): Base dimensionality for the generator's convolutional layers.
            dim_z (int): Dimensionality of the latent space representation.
        """
        super(TextureGenerator, self).__init__()
        self.dim_g = dim_g
        self.dim_z = dim_z

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
        self.conv_9 = nn.ConvTranspose3d(
            dim_g * 2 + dim_z, dim_g, 4, stride=2, padding=1, bias=True
        )
        self.conv_10 = nn.Conv3d(
            dim_g, 3, 3, stride=1, padding=1, bias=True
        )  # Output 3 channels for RGB texture

    def forward(self, voxels, z):
        """
        Forward pass through the generator.

        Processes input data alongside latent space representations through a series of convolutional
        layers to generate texture details. The final output is adjusted to ensure it falls within a
        reasonable range, potentially applying a non-linear activation to map it to a valid color space.

        Parameters:
            voxels (torch.Tensor): Input voxel data.
            z (torch.Tensor): Latent space representation to be integrated at each stage.

        Returns:
            torch.Tensor: The generated texture data with three channels for RGB representation.
        """
        out = voxels

        # Process through convolutional layers, integrating latent z at each step
        for layer in [self.conv_5, self.conv_6, self.conv_7, self.conv_8, self.conv_9]:
            _, _, dimx, dimy, dimz = out.size()
            zs = z.repeat(1, 1, dimx, dimy, dimz)
            out = torch.cat([out, zs], dim=1)
            out = F.leaky_relu(layer(out), negative_slope=0.02, inplace=True)

        # Final texture representation layer
        out = self.conv_10(out)
        out = torch.max(torch.min(out, out * 0.002 + 0.998), out * 0.002)
        out = torch.sigmoid(out)

        return out
