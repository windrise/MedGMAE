import sys
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_cosine_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_scaling_rotation_inverse
import torch.nn.functional as F
from utils.Compute_intensity import compute_intensity, compute_intensity_sparse


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


class GaussianModel:
    def __init__(self, fea_dim=0, with_motion_mask=True, **kwargs):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            # symm = strip_symmetric(actual_covariance)
            # return symm
            return actual_covariance

        # self.active_sh_degree = 0
        # self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._cmesh = torch.empty(0)
        # self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._density = torch.empty(0)
        # self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        self.with_motion_mask = with_motion_mask
        if self.with_motion_mask:
            # Masks stored as features
            fea_dim += 1
        self.fea_dim = fea_dim
        self.feature = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.sigmoid  # torch.exp
        self.scaling_inverse_activation = inverse_sigmoid  # torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def param_names(self):
        return ['_xyz', '_scaling', '_rotation', '_density', 'xyz_gradient_accum']

    @classmethod
    def build_from(cls, gs, **kwargs):
        new_gs = GaussianModel(**kwargs)
        new_gs._xyz = nn.Parameter(gs._xyz)
        # new_gs._features_dc = nn.Parameter(torch.zeros_like(gs._features_dc))
        # new_gs._features_rest = nn.Parameter(torch.zeros_like(gs._features_rest))
        new_gs._scaling = nn.Parameter(gs._scaling)
        new_gs._rotation = nn.Parameter(gs._rotation)
        new_gs._density = nn.Parameter(gs._density)
        new_gs.feature = nn.Parameter(gs.feature)
        # new_gs.max_radii2D = torch.zeros((new_gs.get_xyz.shape[0]), device="cuda")
        return new_gs

    @property
    def motion_mask(self):
        if self.with_motion_mask:
            return torch.sigmoid(self.feature[..., -1:])
        else:
            return torch.ones_like(self._xyz[..., :1])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_rotation_bias(self, rotation_bias=None):
        rotation_bias = rotation_bias if rotation_bias is not None else 0.
        return self.rotation_activation(self._rotation + rotation_bias)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_density(self):
        return self.density_activation(self._density)

    def get_covariance(self, scaling_modifier=1, d_rotation=None, gs_rot_bias=None):
        if d_rotation is not None:
            rotation = quaternion_multiply(self._rotation, d_rotation)
        else:
            rotation = self._rotation
        if gs_rot_bias is not None:
            rotation = rotation / rotation.norm(dim=-1, keepdim=True)
            rotation = quaternion_multiply(gs_rot_bias, rotation)
        return self.covariance_activation(self.get_scaling, scaling_modifier, rotation)

    def get_covariance_phy(self, scaling_modifier=1, d_rotation=None, gs_rot_bias=None):
        if d_rotation is not None:
            rotation = quaternion_multiply(self._rotation, d_rotation)
        else:
            rotation = self._rotation
        if gs_rot_bias is not None:
            rotation = rotation / rotation.norm(dim=-1, keepdim=True)
            rotation = quaternion_multiply(gs_rot_bias, rotation)
        return strip_symmetric(self.covariance_activation(self.get_scaling, scaling_modifier, rotation))

    def get_covariance_inv(self):
        L = build_scaling_rotation_inverse(self.get_scaling, self._rotation)
        actual_covariance_inv = L @ L.transpose(1, 2)
        return actual_covariance_inv

    # ... Rest of the GaussianModel class implementation ...
    # Only showing the essential parts for brevity

    def state_dict(self):
        return {
            '_xyz': self._xyz,
            # '_sigma': self._sigma,
            '_density': self._density,
            '_scaling': self._scaling,
            '_rotation': self._rotation,
        }

    def load_state_dict(self, state_dict):
        self._xyz = nn.Parameter(state_dict['_xyz'].clone().detach().requires_grad_(True))
        self._density = nn.Parameter(state_dict['_density'].clone().detach().requires_grad_(True))
        self._scaling = nn.Parameter(state_dict['_scaling'].clone().detach().requires_grad_(True))
        self._rotation = nn.Parameter(state_dict['_rotation'].clone().detach().requires_grad_(True))


class StandardGaussianModel(GaussianModel):
    def __init__(self, fea_dim=0, with_motion_mask=True, all_the_same=False):
        super().__init__(fea_dim, with_motion_mask)
        self.all_the_same = all_the_same

    @property
    def get_scaling(self):
        scaling = self._scaling.mean()[None, None].expand_as(
            self._scaling) if self.all_the_same else self._scaling.mean(dim=1, keepdim=True).expand_as(self._scaling)
        return self.scaling_activation(scaling)


def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                             torch.linspace(0, 1, steps=h), \
                                             torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid


def get_unmasked_grid_points(image_size, pixel_mask):
    """
    Determine the coordinates of 3D grid points to compute based on the pixel_mask.

    Args:
        image_size (tuple): (c, h, w)
        pixel_mask (torch.Tensor): 3D mask [B, 1, C, H, W], where 1 indicates points to compute

    Returns:
        torch.Tensor: Unmasked 3D grid point coordinates [M, 3], range [0, 1]
        torch.Tensor: Indices of these points in the original dense grid [M, 3]
    """
    c, h, w = image_size
    device = pixel_mask.device
    
    if pixel_mask.dim() == 5 and pixel_mask.shape[1] == 1:  # [B, 1, C, H, W]
        mask_3d = pixel_mask[0, 0]  # [C, H, W]
    else:
        raise ValueError(f"Expected pixel_mask with shape [B, 1, C, H, W], got {pixel_mask.shape}")
    
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, steps=c, device=device),
        torch.linspace(0, 1, steps=h, device=device),
        torch.linspace(0, 1, steps=w, device=device),
        indexing='ij'
    )
    grid_coords = torch.stack([grid_z, grid_y, grid_x], dim=-1) # [c, h, w, 3]
    
    unmasked_indices = torch.nonzero(mask_3d, as_tuple=False) # [M, 3] indices (z, y, x)
    
    unmasked_grid_points = grid_coords[unmasked_indices[:, 0], unmasked_indices[:, 1], unmasked_indices[:, 2]] # [M, 3]
    
    return unmasked_grid_points, unmasked_indices


def rendering(pc: GaussianModel, image_size):
    """
    Render Gaussians from a GaussianModel to an image
    
    Args:
        pc (GaussianModel): Gaussian model with positions, scales, rotations, and densities
        image_size (tuple): Size of the output image (c, h, w)
        
    Returns:
        torch.Tensor: Rendered image in the format [1, c, h, w, 1]
    """
    # Create 3D grid
    img_dim = image_size
    grid = create_grid_3d(*img_dim)
    grid = grid.cuda()
    grid = grid.unsqueeze(0).repeat(1, 1, 1, 1, 1)

    grid_point = grid.unsqueeze(-2)  # [batchsize, z, x, y, 1, 3]
    z, x, y = grid_point.shape[1:4]
    density_grid = torch.zeros(1, z, x, y, 1, device='cuda', requires_grad=True)
    
    # Get inverse covariance
    inv_covariance = pc.get_covariance_inv()
    
    # Compute intensity
    density_grid = compute_intensity(
        pc.get_xyz.contiguous(),
        grid_point.contiguous(),
        pc.get_density.contiguous(),
        inv_covariance.contiguous(),
        pc.get_scaling.contiguous(),
        density_grid.contiguous())

    return density_grid


def rendering_sparse(pc: GaussianModel, image_size, pixel_mask):
    """
    Sparse rendering function, only computes unmasked points

    Args:
        pc (GaussianModel): Gaussian model
        image_size (tuple): (c, h, w)
        pixel_mask (torch.Tensor): 3D mask [B, 1, C, H, W], where 1 indicates points to compute

    Returns:
        torch.Tensor: Rendering result, dense tensor of shape [1, c, h, w, 1]
    """
    c, h, w = image_size
    device = pc.get_xyz.device
    
    unmasked_grid_points, unmasked_indices = get_unmasked_grid_points(image_size, pixel_mask)
    
    if unmasked_grid_points.shape[0] == 0:
        return torch.zeros((1, c, h, w, 1), device=device)
    
    xyz = pc.get_xyz.contiguous()
    density = pc.get_density.contiguous()
    scaling = pc.get_scaling.contiguous()
    inv_covariance = pc.get_covariance_inv().contiguous()
    
    num_unmasked_points = unmasked_grid_points.shape[0]
    sparse_intensity_values = torch.zeros(num_unmasked_points, 1, device=device, requires_grad=True)
    
    sparse_intensity_values = compute_intensity_sparse(
        xyz,
        unmasked_grid_points.contiguous(),
        density,
        inv_covariance,
        scaling,
        sparse_intensity_values
    )
    
    dense_output = torch.zeros((1, c, h, w, 1), device=device)
    dense_output[0, unmasked_indices[:, 0], unmasked_indices[:, 1], unmasked_indices[:, 2], 0] = sparse_intensity_values.view(-1)
    
    return dense_output


def rendering2(xyz, scaling, rotation, density, image_size):
    """
    Render Gaussians directly from parameters to an image without creating a GaussianModel
    
    Args:
        xyz (torch.Tensor): Positions of Gaussians [N, 3]
        scaling (torch.Tensor): Scales of Gaussians [N, 3]
        rotation (torch.Tensor): Rotations as quaternions [N, 4]
        density (torch.Tensor): Densities of Gaussians [N, 1]
        image_size (tuple): Size of the output image (c, h, w)
        
    Returns:
        torch.Tensor: Rendered image in the format [1, z, x, y, 1]
    """
    # Ensure all inputs are float32 to avoid Half precision errors
    xyz = xyz.float()
    scaling = scaling.float()
    rotation = rotation.float()
    density = density.float()
    
    # Create 3D grid
    img_dim = image_size
    grid = create_grid_3d(*img_dim)
    grid = grid.cuda()
    grid = grid.unsqueeze(0).repeat(1, 1, 1, 1, 1)

    grid_point = grid.unsqueeze(-2)  # [batchsize, z, x, y, 1, 3]
    z, x, y = grid_point.shape[1:4]
    density_grid = torch.zeros(1, z, x, y, 1, device='cuda', dtype=torch.float32, requires_grad=True)
    
    # Apply activations to parameters
    # activated_scaling = torch.sigmoid(scaling)
    activated_scaling = scaling
    #activated_rotation = F.normalize(rotation, dim=-1)
    activated_rotation = rotation
    # activated_density = torch.sigmoid(density)
    activated_density = density

    # Compute inverse covariance
    L = build_scaling_rotation_inverse(activated_scaling, activated_rotation)
    inv_covariance = L @ L.transpose(1, 2)
    
    try:
        # Compute intensity
        density_grid = compute_intensity(
            xyz.contiguous(),
            grid_point.contiguous(),
            0.1*activated_density.contiguous(),
            inv_covariance.contiguous(),
            activated_scaling.contiguous(),
            density_grid.contiguous())
    except Exception as e:
        print(f"Error rendering: {e}")
        print(f"Positions shape: {xyz.shape}, dtype: {xyz.dtype}")
        print(f"Scales shape: {activated_scaling.shape}, dtype: {activated_scaling.dtype}")
        print(f"Rotations shape: {activated_rotation.shape}, dtype: {activated_rotation.dtype}")
        print(f"Densities shape: {activated_density.shape}, dtype: {activated_density.dtype}")
        raise e

    return density_grid


def rendering2_sparse(xyz, scaling, rotation, density, image_size, pixel_mask):
    """
    Sparse version of direct parameter rendering

    Args:
        xyz (torch.Tensor): Gaussian center coordinates [N, 3]
        scaling (torch.Tensor): Scaling parameters [N, 3]
        rotation (torch.Tensor): Rotation parameters [N, 4]
        density (torch.Tensor): Density parameters [N, 1]
        image_size (tuple): (c, h, w)
        pixel_mask (torch.Tensor): 3D mask [B, 1, C, H, W], where 1 indicates points to compute

    Returns:
        torch.Tensor: Rendering result, dense tensor of shape [1, c, h, w, 1]
    """
    c, h, w = image_size
    device = xyz.device
    
    unmasked_grid_points, unmasked_indices = get_unmasked_grid_points(image_size, pixel_mask)
    
    if unmasked_grid_points.shape[0] == 0:
        return torch.zeros((1, c, h, w, 1), device=device)
    
    xyz = xyz.float().contiguous()
    scaling = scaling.float().contiguous()
    rotation = rotation.float().contiguous()
    density = density.float().contiguous()
    unmasked_grid_points = unmasked_grid_points.float().contiguous()
    
    num_unmasked_points = unmasked_grid_points.shape[0]
    sparse_intensity_values = torch.zeros(num_unmasked_points, 1, device=device, dtype=torch.float32, requires_grad=True)
    
    L = build_scaling_rotation_inverse(scaling, rotation)
    inv_covariance = (L @ L.transpose(1, 2)).contiguous()
    
    sparse_intensity_values = compute_intensity_sparse(
        xyz,
        unmasked_grid_points,
        0.1 * density,
        inv_covariance,
        scaling,
        sparse_intensity_values
    )
    
    dense_output = torch.zeros((1, c, h, w, 1), device=device)
    dense_output[0, unmasked_indices[:, 0], unmasked_indices[:, 1], unmasked_indices[:, 2], 0] = sparse_intensity_values.view(-1)
    
    return dense_output


def batch_rendering2(positions, scales, rotations, densities, image_size):
    """
    Render a batch of Gaussian models
    
    Args:
        positions (torch.Tensor): Batch of positions [B, N, 3]
        scales (torch.Tensor): Batch of scales [B, N, 3]
        rotations (torch.Tensor): Batch of rotations [B, N, 4]
        densities (torch.Tensor): Batch of densities [B, N, 1]
        image_size (tuple): Size of the output image (c, h, w)
        
    Returns:
        torch.Tensor: Batch of rendered images [B, 1, c, h, w]
    """
    # Ensure all inputs are float32 to avoid Half precision errors
    positions = positions.float()
    scales = scales.float()
    rotations = rotations.float()
    densities = densities.float()
    
    batch_size = positions.shape[0]
    rendered_images = []
    
    # Process each item in the batch
    for b in range(batch_size):
        try:
            # Render the current batch item
            rendered = rendering2(
                positions[b], 
                scales[b], 
                rotations[b], 
                densities[b], 
                image_size
            )
            
            # Check the shape of rendered and reshape accordingly
            # Handle the case where rendered is [1, z, x, y, 1]
            c, h, w = image_size
            
            # Print rendered shape for debugging
            # if b == 0:  # Only print for first batch to avoid spam
            #     print(f"Rendered shape: {rendered.shape}")
            
            # Proper reshaping based on the actual dimensions of rendered
            if rendered.dim() == 5:  # [1, z, x, y, 1]
                rendered = rendered.reshape(1, 1, c, h, w)
            elif rendered.dim() == 4:  # [1, z, x, y]
                rendered = rendered.unsqueeze(1)  # Make it [1, 1, z, x, y]
            else:
                print(f"Warning: Unexpected rendered shape: {rendered.shape}")
                # Create a placeholder if needed
                rendered = rendered.reshape(1, 1, c, h, w)
                
            rendered_images.append(rendered)
            
        except Exception as e:
            print(f"Error rendering batch item {b}: {e}")
            print(f"Positions shape: {positions[b].shape}")
            print(f"Scales shape: {scales[b].shape}")
            print(f"Rotations shape: {rotations[b].shape}")
            print(f"Densities shape: {densities[b].shape}")
            
            # Create a placeholder empty rendered image
            c, h, w = image_size
            empty_rendered = torch.zeros((1, 1, c, h, w), device=positions.device, dtype=torch.float32)
            rendered_images.append(empty_rendered)
    
    # Stack along batch dimension
    rendered_batch = torch.cat(rendered_images, dim=0)
    
    return rendered_batch


def batch_rendering2_sparse(positions, scales, rotations, densities, image_size, pixel_masks):
    """
    Sparse rendering of a batch of Gaussian models

    Args:
        positions (torch.Tensor): Batch positions [B, N, 3]
        scales (torch.Tensor): Batch scales [B, N, 3]
        rotations (torch.Tensor): Batch rotations [B, N, 4]
        densities (torch.Tensor): Batch densities [B, N, 1]
        image_size (tuple): Output image size (c, h, w)
        pixel_masks (torch.Tensor): Batch masks [B, 1, C, H, W], where 1 indicates points to compute

    Returns:
        torch.Tensor: Batch rendered images [B, 1, c, h, w]
    """
    positions = positions.float()
    scales = scales.float()
    rotations = rotations.float()
    densities = densities.float()
    
    batch_size = positions.shape[0]
    rendered_images = []
    
    for b in range(batch_size):
        try:
            rendered = rendering2_sparse(
                positions[b], 
                scales[b], 
                rotations[b], 
                densities[b], 
                image_size,
                pixel_masks[b:b+1]
            )
            
            c, h, w = image_size
            
            if rendered.dim() == 5:  # [1, z, x, y, 1]
                rendered = rendered.reshape(1, 1, c, h, w)
            elif rendered.dim() == 4:  # [1, z, x, y]
                rendered = rendered.unsqueeze(1)
            else:
                print(f"Warning: Unexpected rendered shape: {rendered.shape}")
                rendered = rendered.reshape(1, 1, c, h, w)
                
            rendered_images.append(rendered)
            
        except Exception as e:
            print(f"Error rendering batch item {b}: {e}")
            print(f"Positions shape: {positions[b].shape}")
            print(f"Scales shape: {scales[b].shape}")
            print(f"Rotations shape: {rotations[b].shape}")
            print(f"Densities shape: {densities[b].shape}")
            
            c, h, w = image_size
            empty_rendered = torch.zeros((1, 1, c, h, w), device=positions.device, dtype=torch.float32)
            rendered_images.append(empty_rendered)
    
    rendered_batch = torch.cat(rendered_images, dim=0)
    
    return rendered_batch


def sparse_render_to_patches(positions, scales, rotations, densities, input_size, pixel_mask, msk_length, patch_size, in_chans=1):
    """
    Sparse render Gaussian point cloud and reshape to patch format matching msk_x

    Args:
        positions (torch.Tensor): Gaussian position parameters [B, N, 3]
        scales (torch.Tensor): Gaussian scale parameters [B, N, 3]
        rotations (torch.Tensor): Gaussian rotation parameters [B, N, 4]
        densities (torch.Tensor): Gaussian density parameters [B, N, 1]
        input_size (tuple): Input image shape (C, H, W, D)
        pixel_mask (torch.Tensor): Pixel mask indicating regions to render [B, C, H, W, D]
        msk_length (int): Number of masked patches
        patch_size (tuple): Patch size (ph, pw, pd)
        in_chans (int): Number of input channels, default 1

    Returns:
        torch.Tensor: Reshaped sparse rendering result matching msk_x shape [B, msk_length, ph*pw*pd*in_chans]
    """
    batch_size = positions.shape[0]
    device = positions.device
    
    positions_f32 = positions.float().contiguous()
    scales_f32 = scales.float().contiguous()
    rotations_f32 = rotations.float().contiguous()
    densities_f32 = densities.float().contiguous()
    
    all_unmasked_points = []
    all_batch_indices = []
    
    for b in range(batch_size):
        unmasked_points, _ = get_unmasked_grid_points(input_size, pixel_mask[b:b+1])
        if unmasked_points.shape[0] > 0:
            all_unmasked_points.append(unmasked_points)
            all_batch_indices.append(torch.full((unmasked_points.shape[0],), b, device=device))
    
    if not all_unmasked_points:
        pixels_per_patch = np.prod(patch_size) * in_chans
        return torch.zeros(batch_size, msk_length, pixels_per_patch, device=device)
    
    all_points = torch.cat(all_unmasked_points, dim=0)
    batch_indices = torch.cat(all_batch_indices, dim=0)
    
    all_batch_results = []

    for b in range(batch_size):
        batch_mask = batch_indices == b
        if not batch_mask.any():
            continue
        
        current_points = all_points[batch_mask]
        
        L = build_scaling_rotation_inverse(scales_f32[b], rotations_f32[b])
        inv_covariance = (L @ L.transpose(1, 2)).contiguous()
        
        current_values = torch.zeros(current_points.shape[0], 1, device=device, dtype=torch.float32)
        
        current_values = compute_intensity_sparse(
            positions_f32[b],
            current_points,
            0.1 * densities_f32[b],
            inv_covariance,
            scales_f32[b],
            current_values
        )
        
        all_batch_results.append((b, current_values))
    
    pixels_per_patch = np.prod(patch_size) * in_chans
    pixels_per_batch = msk_length * pixels_per_patch
    
    batch_values = []
    current_idx = 0
    
    for b in range(batch_size):
        batch_mask = batch_indices == b
        num_points = batch_mask.sum().item()
        
        if num_points > 0:
            batch_values.append(all_batch_results[current_idx][1])
        else:
            batch_values.append(torch.zeros(0, 1, device=device))
        
        current_idx += 1
    
    recon_x = torch.zeros(batch_size, msk_length, pixels_per_patch, device=device)
    
    for b in range(batch_size):
        batch_points = batch_values[b]
        if batch_points.shape[0] == pixels_per_batch:
            recon_x[b] = batch_points.reshape(msk_length, pixels_per_patch)
        elif batch_points.shape[0] > 0:
            print(f"Warning: Pixel count mismatch for batch {b}, expected {pixels_per_batch}, got {batch_points.shape[0]}")
            if batch_points.shape[0] < pixels_per_batch:
                padding = torch.zeros(pixels_per_batch - batch_points.shape[0], 1, device=device)
                padded = torch.cat([batch_points, padding], dim=0)
                recon_x[b] = padded.reshape(msk_length, pixels_per_patch)
            else:
                recon_x[b] = batch_points[:pixels_per_batch].reshape(msk_length, pixels_per_patch)
    
    return recon_x


