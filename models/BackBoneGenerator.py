import torch
import torch.nn as nn
import torch.nn.functional as F


class BackBoneGenerator(nn.Module):
    """
    A generator module that adopts a dual branch small architecture, expanding the input dimensionality by a factor of 8.

    This generator integrates additional dimensions into its processing pipeline, progressively increasing
    the depth while maintaining spatial dimensions through dilated convolutions. It's specifically designed to
    augment input data with latent representations at multiple stages of the processing pipeline.

    Attributes:
        dim_g (int): Base dimensionality for the generator's convolutional layers.
        dim_z (int): Dimensionality of the shape latent space representation.
    """

    def __init__(self, dim_g, dim_z):
        """
        Initializes the BackBoneGenerator model with the specified dimensions.

        Parameters:
            dim_g (int): Base dimensionality for the generator's convolutional layers.
            dim_z (int): Dimensionality of the latent space representation.
        """
        super(BackBoneGenerator, self).__init__()
        self.dim_g = dim_g
        self.dim_z = dim_z

        # Define the convolutional layers, including dilated convolutions to control the receptive field size
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv3d(
                    1 + dim_z, dim_g, 3, stride=1, dilation=1, padding=1, bias=True
                ),
                nn.Conv3d(
                    dim_g + dim_z,
                    dim_g * 2,
                    3,
                    stride=1,
                    dilation=2,
                    padding=2,
                    bias=True,
                ),
                nn.Conv3d(
                    dim_g * 2 + dim_z,
                    dim_g * 4,
                    3,
                    stride=1,
                    dilation=2,
                    padding=2,
                    bias=True,
                ),
                nn.Conv3d(
                    dim_g * 4 + dim_z,
                    dim_g * 8,
                    3,
                    stride=1,
                    dilation=1,
                    padding=1,
                    bias=True,
                ),
                nn.Conv3d(
                    dim_g * 8 + dim_z,
                    dim_g * 8,
                    3,
                    stride=1,
                    dilation=1,
                    padding=1,
                    bias=True,
                ),
            ]
        )

    def forward(self, voxels, z, is_training=False):
        """
        Forward pass through the generator.

        Processes input data alongside latent space representations through a series of convolutional layers,
        leveraging dilated convolutions and leaky ReLU activations to generate output data.

        Parameters:
            voxels (torch.Tensor): Input voxel data.
            z (torch.Tensor): Latent space representation to be integrated at each convolutional layer.
            is_training (bool, optional): Flag indicating if the model is in training mode. Defaults to False.

        Returns:
            torch.Tensor: The output of the generator after processing.
        """
        out = voxels

        # Repeating and concatenating z with the output at each convolutional layer
        for conv_layer in self.conv_layers:
            _, _, dimx, dimy, dimz = out.size()
            zs = z.repeat(1, 1, dimx, dimy, dimz)
            out = torch.cat([out, zs], dim=1)
            out = conv_layer(out)
            out = F.leaky_relu(out, 0.02, inplace=True)

        return out
