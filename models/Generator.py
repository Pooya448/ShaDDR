import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BackBoneGenerator import BackBoneGenerator
from models.GeometryGenerator import GeometryGenerator
from models.TextureGenerator import TextureGenerator


class Generator(nn.Module):
    """
    A generator module that combines geometry and texture synthesis into a dual-pathway architecture,
    designed for generating detailed 3D models with both structural and surface details.

    This class orchestrates the generation process by utilizing separate backbone generators for geometry and texture,
    each tailored to process input voxel data for their respective synthesis tasks. Depending on the mode of operation,
    it either generates geometry, texture, or both in sequence.

    Attributes:
        dim_g (int): Base dimensionality for the generator's convolutional layers.
        dim_prob (int): Dimensionality related to the probabilistic aspects of generation.
        dim_z (int): Dimensionality of the latent space representation.
        geometry_codes (nn.Parameter): Learnable parameters representing geometry-specific latent codes.
        texture_codes (nn.Parameter): Learnable parameters representing texture-specific latent codes.
        geometry_backbone_generator (BackBoneGenerator): The backbone generator for initial voxel processing for geometry.
        texture_backbone_generator (BackBoneGenerator): The backbone generator for initial voxel processing for texture.
        geometry_generator (GeometryGenerator): The geometry generator for structural synthesis.
        texture_generator (TextureGenerator): The texture generator for surface detail synthesis.
    """

    def __init__(self, dim_g, dim_z, dim_prob):
        """
        Initializes the Generator model with the specified dimensions.

        Parameters:
            dim_g (int): Base dimensionality for the generator's convolutional layers.
            dim_prob (int): Dimensionality related to the probabilistic aspects of generation.
            dim_z (int): Dimensionality of the latent space representation.
        """
        super(Generator, self).__init__()
        self.dim_g = dim_g
        self.dim_prob = dim_prob
        self.dim_z = dim_z

        # Initialize geometry and texture codes
        geometry_codes = torch.zeros((dim_prob, dim_z))
        self.geometry_codes = nn.Parameter(geometry_codes)
        nn.init.constant_(self.geometry_codes, 0.0)

        texture_codes = torch.zeros((dim_prob, dim_z))
        self.texture_codes = nn.Parameter(texture_codes)
        nn.init.constant_(self.texture_codes, 0.0)

        # Instantiate separate backbone generators for geometry and texture
        self.geometry_backbone_generator = BackBoneGenerator(dim_g, dim_z)
        self.texture_backbone_generator = BackBoneGenerator(dim_g, dim_z)

        # Instantiate the geometry and texture generators
        self.geometry_generator = GeometryGenerator(dim_g, dim_z)
        self.texture_generator = TextureGenerator(dim_g, dim_z)

    def forward(self, voxels, geometry_z, texture_z, mask, is_geometry_training=True):
        """
        Forward pass through the generator.

        Depending on the training phase, it either generates geometry, texture, or both. For geometry
        generation, it processes input voxels through the geometry backbone and geometry pathways. For texture
        generation, it detaches geometry outcomes and processes through the texture backbone and texture pathway.

        Parameters:
            voxels (torch.Tensor): Input voxel data.
            geometry_z (torch.Tensor): Latent space representation for geometry generation.
            texture_z (torch.Tensor): Latent space representation for texture generation.
            mask (torch.Tensor): Mask indicating regions of interest for generation.
            is_geometry_training (bool, optional): Flag to control the mode of generation. Defaults to True.

        Returns:
            tuple: Generated geometry and/or texture depending on the mode of operation.
        """
        out = voxels

        if is_geometry_training:
            out = self.geometry_backbone_generator(out, geometry_z)
            out_256, out_128 = self.geometry_generator(out, geometry_z, mask)

            return out_256, out_128
        else:
            with torch.no_grad():
                out_geometry = self.geometry_backbone_generator(out, geometry_z)
                out_geometry, _ = self.geometry_generator(
                    out_geometry, geometry_z, mask
                )

            out_texture = self.texture_backbone_generator(out, texture_z)
            out_texture = self.texture_generator(out_texture, texture_z)

            return out_geometry, out_texture
