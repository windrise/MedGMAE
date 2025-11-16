import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_source_path = os.path.join(current_dir, 'discretize_grid.cu')

compute_intensity_cuda = load(
    name='compute_intensity_cuda', 
    sources=[cuda_source_path],
    extra_cflags=['-O3', '-std=c++14'],
    extra_cuda_cflags=['-O3'],
    verbose=True
)


class IntensityComputation(Function):
    @staticmethod
    def forward(ctx, gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
        ctx.save_for_backward(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid)
        
        intensity_grid = compute_intensity_cuda.compute_intensity(
            gaussian_centers,
            grid_points,
            intensities,
            inv_covariances,
            scalings,
            intensity_grid
        )
        
        return intensity_grid

    @staticmethod
    def backward(ctx, grad_output):
        gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid = ctx.saved_tensors
        
        grad_gaussian_centers, grad_intensities, grad_inv_covariances, grad_intensity_grid = compute_intensity_cuda.compute_intensity_backward(
            grad_output,
            gaussian_centers,
            grid_points,
            intensities,
            inv_covariances,
            scalings,
            intensity_grid
        )

        return grad_gaussian_centers, None, grad_intensities, grad_inv_covariances, None, grad_intensity_grid

def compute_intensity(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
    """
    Compute Gaussian distribution intensity. Supports mixed precision (fp16 and fp32).

    Args:
        gaussian_centers: Gaussian center point coordinates [N, 3]
        grid_points: Grid point coordinates [batchsize, z, x, y, 1, 3]
        intensities: Gaussian intensities [N, 1]
        inv_covariances: Inverse Gaussian covariance matrix [N, 9]
        scalings: Gaussian scaling parameters [N, 3]
        intensity_grid: Output intensity grid [1, z, x, y, 1]

    Returns:
        Updated intensity grid [1, z, x, y, 1]
    """
    return IntensityComputation.apply(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid)


class SparseIntensityComputation(Function):
    @staticmethod
    def forward(ctx, gaussian_centers, sparse_grid_points, intensities, inv_covariances, scalings, sparse_intensity_values_out):
        num_sparse_points = sparse_grid_points.shape[0]
        if num_sparse_points == 0:
            return torch.zeros_like(sparse_intensity_values_out) # Handle empty input
            
        computed_intensities = compute_intensity_cuda.compute_intensity_sparse_forward(
            gaussian_centers,
            sparse_grid_points, # [M, 3]
            intensities,
            inv_covariances,
            scalings,
            num_sparse_points
        )
        
        ctx.save_for_backward(gaussian_centers, sparse_grid_points, intensities, inv_covariances, scalings, computed_intensities)
        return computed_intensities # [M, 1]
        
    @staticmethod
    def backward(ctx, grad_output): # grad_output shape [M, 1]
        gaussian_centers, sparse_grid_points, intensities, inv_covariances, scalings, computed_intensities = ctx.saved_tensors
        num_sparse_points = sparse_grid_points.shape[0]
        
        if num_sparse_points == 0:
            # Return gradients of correct shape and type, all zeros
            return torch.zeros_like(gaussian_centers), \
                   None, \
                   torch.zeros_like(intensities), \
                   torch.zeros_like(inv_covariances), \
                   None, \
                   None
                   
        grad_gaussian_centers, grad_intensities, grad_inv_covariances = compute_intensity_cuda.compute_intensity_sparse_backward(
            grad_output.contiguous(), # [M, 1]
            gaussian_centers,
            sparse_grid_points, # [M, 3]
            intensities,
            inv_covariances,
            scalings,
            computed_intensities,
            num_sparse_points
        )
        
        return grad_gaussian_centers, None, grad_intensities, grad_inv_covariances, None, None

def compute_intensity_sparse(gaussian_centers, sparse_grid_points, intensities, inv_covariances, scalings, sparse_intensity_values_out):
    """
    Compute Gaussian intensity on sparse grid points.

    Args:
        gaussian_centers: Gaussian center coordinates [N, 3]
        sparse_grid_points: Sparse point coordinates to compute [M, 3]
        intensities: Gaussian intensities [N, 1]
        inv_covariances: Inverse covariance matrix [N, 9]
        scalings: Scaling parameters [N, 3]
        sparse_intensity_values_out: Output value placeholder [M, 1]

    Returns:
        torch.Tensor: Computed sparse point intensity values [M, 1]
    """
    return SparseIntensityComputation.apply(gaussian_centers, sparse_grid_points, intensities, inv_covariances, scalings, sparse_intensity_values_out)
