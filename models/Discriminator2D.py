import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator2D(nn.Module):
    """
    A 2D discriminator model designed specifically for a receptive field (RF) of 18.

    This version of the discriminator processes input images through a series of convolutional layers
    configured to work optimally with the specified RF size, extracting features for discrimination tasks.

    Attributes:
        dim_d (int): The dimensionality of the first layer's output.
        dim_z (int): The output dimensionality of the final layer.
    """

    def __init__(self, dim_d, dim_z, dim_in=1):
        """
        Initializes the Discriminator2D model with the specified dimensions for an RF of 18.

        Parameters:
            dim_d (int): Dimensionality of the first layer's output.
            dim_z (int): Output dimensionality of the final layer.
            dim_in (int, optional): Number of input channels. Defaults to 1.
        """
        super(Discriminator2D, self).__init__()
        self.dim_d = dim_d
        self.dim_z = dim_z

        # Define the sequence of convolutional layers for RF of 18
        self.conv_layers = nn.Sequential(
            nn.Conv2d(dim_in, dim_d, 4, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(dim_d, dim_d * 2, 3, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(dim_d * 2, dim_d * 4, 3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(dim_d * 4, dim_d * 8, 3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(dim_d * 8, dim_d * 16, 3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(dim_d * 16, dim_z, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, voxels, is_training=False):
        """
        Forward pass through the discriminator.

        Processes input images through the defined sequence of convolutions and activations to produce a feature vector.

        Parameters:
            voxels (torch.Tensor): The input image data.
            is_training (bool, optional): Flag indicating if the model is in training mode. Defaults to False.

        Returns:
            torch.Tensor: The discriminator's output.
        """
        return self.conv_layers(voxels)
