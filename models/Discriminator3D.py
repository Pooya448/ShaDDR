import torch.nn as nn


class Discriminator3D(nn.Module):
    """
    A discriminator model with a receptive field of 18.

    This module uses a series of 3D convolutions to process input voxel data. The architecture
    is designed to progressively increase the depth while reducing the spatial dimensions of the
    input tensor, culminating in a feature vector suited for discrimination tasks.

    Attributes:
        dim_d (int): Dimensionality of the initial layer's output.
        dim_z (int): Output dimensionality of the final layer.
        conv_layers (nn.Sequential): Sequential container of all convolutional layers.
    """

    def __init__(self, dim_d, dim_z, dim_in=1):
        """
        Initializes the Discriminator3D model with specified dimensions.

        Parameters:
            dim_d (int): Dimensionality of the initial layer's output.
            dim_z (int): Output dimensionality of the final layer.
            dim_in (int, optional): Number of input channels. Defaults to 1.
        """
        super(Discriminator3D, self).__init__()
        self.dim_d = dim_d
        self.dim_z = dim_z

        # Define convolutional layers with appropriate dimensions and activations
        self.conv_layers = nn.Sequential(
            nn.Conv3d(dim_in, dim_d, 4, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv3d(dim_d, dim_d * 2, 3, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv3d(dim_d * 2, dim_d * 4, 3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv3d(dim_d * 4, dim_d * 8, 3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv3d(dim_d * 8, dim_d * 16, 3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv3d(dim_d * 16, dim_z, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, voxels):
        """
        Forward pass of the discriminator.

        Processes input voxel data through a series of convolutions to produce a discrimination
        output.

        Parameters:
            voxels (torch.Tensor): Input voxel data.

        Returns:
            torch.Tensor: The output of the discriminator.
        """
        return self.conv_layers(voxels)
