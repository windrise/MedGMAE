import torch
import numpy as np
from timm.models.layers.helpers import to_3tuple
import matplotlib.pyplot as plt

def patches3d_to_grid(tensor, patch_size, grid_size, in_chans=1, hidden_axis='d'):
    """
    Convert 3D patches to a 2D grid for visualization.
    
    Args:
        tensor: tensor of shape [B, N, C], B is batch size, N is number of patches, C is channels*p*p*p
        patch_size: size of each patch (can be an int or tuple)
        grid_size: size of the grid (can be an int or tuple)
        in_chans: number of input channels
        hidden_axis: which axis to hide (d, h, or w)
        
    Returns:
        2D grid image for visualization
    """
    try:
        print(f"\npatches3d_to_grid input: ")
        print(f"  - tensor.shape: {tensor.shape}")
        print(f"  - patch_size: {patch_size}")
        print(f"  - grid_size: {grid_size}")
        print(f"  - in_chans: {in_chans}")
        print(f"  - hidden_axis: {hidden_axis}")
        
        B, N, C = tensor.shape
        pd, ph, pw = patch_size
        gh, gw, gd = grid_size
        
        if N != gh*gw*gd:
            print(f"\nWarning: N = {N} != gh*gw*gd = {gh*gw*gd}, may cause visualization errors")
        
        tensor = tensor.reshape(B, gh, gw, gd, -1)
        
        patch_volume = pd * ph * pw * in_chans
        if tensor.shape[4] != patch_volume:
            print(f"\nWarning: patch size mismatch: {tensor.shape[4]} != {patch_volume}")
            actual_volume = tensor.shape[4]
            cube_side = int(pow(actual_volume / in_chans, 1/3) + 0.5)
            pd = ph = pw = cube_side
            print(f"  - Auto-adjusted patch size to: {pd}x{ph}x{pw}")
        
        try:
            tensor = tensor.reshape(B, gh, gw, gd, pd, ph, pw, in_chans)
        except RuntimeError as e:
            print(f"\nReshape failed: {e}")
            print(f"  - tensor.shape: {tensor.shape}, B={B}, gh={gh}, gw={gw}, gd={gd}, pd={pd}, ph={ph}, pw={pw}, in_chans={in_chans}")
            print(f"  - Expected shape: [{B}, {gh}, {gw}, {gd}, {pd}, {ph}, {pw}, {in_chans}]")
            print(f"  - Attempting auto-adjustment...")

            actual_elements = tensor.shape[4]
            adjusted_pd = adjusted_ph = adjusted_pw = int(pow(actual_elements / in_chans, 1/3))

            try:
                tensor = tensor.reshape(B, gh, gw, gd, adjusted_pd, adjusted_ph, adjusted_pw, in_chans)
                pd, ph, pw = adjusted_pd, adjusted_ph, adjusted_pw
                print(f"  - Successfully adjusted to: pd={pd}, ph={ph}, pw={pw}")
            except RuntimeError:
                print("  - Note: Returning average value image")
                simple_grid = np.zeros((B*50, gh*50), dtype=np.float32) + 0.5
                return simple_grid
        
        if hidden_axis == 'd':
            middle_d = pd // 2
            tensor = tensor[:, :, :, :, middle_d, :, :, :]  # [B, gh, gw, gd, ph, pw, C]
            
            tensor = tensor.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, gh*ph, gw*pw*gd, in_chans)
        elif hidden_axis == 'h':
            middle_h = ph // 2
            tensor = tensor[:, :, :, :, :, middle_h, :, :]  # [B, gh, gw, gd, pd, pw, C]
            
            tensor = tensor.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, gh*pd, gw*pw*gd, in_chans)
        elif hidden_axis == 'w':
            middle_w = pw // 2
            tensor = tensor[:, :, :, :, :, :, middle_w, :]  # [B, gh, gw, gd, pd, ph, C]
            
            tensor = tensor.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, gh*pd, gw*ph*gd, in_chans)
        else:
            raise ValueError(f"Invalid hidden_axis: {hidden_axis}. Must be 'd', 'h', or 'w'.")
        
        grid = tensor.cpu().detach().numpy()
        
        if in_chans == 1:
            grid = grid.squeeze(-1)
        
        grid = np.concatenate([grid[i] for i in range(grid.shape[0])], axis=0)

        print(f"  - Final output grid shape: {grid.shape}")
        return grid
    except Exception as e:
        import traceback
        print(f"\nGeneral error in patches3d_to_grid: {e}")
        traceback.print_exc()
        return np.ones((100, 100), dtype=np.float32) * 0.5

def visualize_tensor_slices(tensor, slices_per_row=4, depth_indices=None, fig_size=(20, 15)):
    """
    Visualize slices from a 3D tensor as a grid of 2D images.
    
    Args:
        tensor: tensor of shape [B, C, D, H, W] or [C, D, H, W]
        slices_per_row: number of slices to show per row
        depth_indices: list of depth indices to visualize (if None, evenly spaced slices are chosen)
        fig_size: figure size for the plot
        
    Returns:
        matplotlib figure object
    """
    # Ensure tensor is on CPU and convert to numpy
    tensor = tensor.detach().cpu().numpy()
    
    # Add batch dimension if not present
    if len(tensor.shape) == 4:
        tensor = tensor[np.newaxis, ...]
    
    B, C, D, H, W = tensor.shape
    
    # Select depth indices if not provided
    if depth_indices is None:
        num_slices = min(8, D)
        depth_indices = np.linspace(0, D-1, num_slices, dtype=int)
    
    # Calculate the number of rows needed
    num_slices = len(depth_indices)
    num_rows = (B * num_slices + slices_per_row - 1) // slices_per_row
    
    fig, axes = plt.subplots(num_rows, slices_per_row, figsize=fig_size)
    
    # Make axes accessible as a 2D array (even if there's only 1 row)
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    elif slices_per_row == 1:
        axes = np.expand_dims(axes, axis=1)
        
    # Flatten for easy access
    all_axes = axes.flatten()
    
    # Counter for the number of plots created
    count = 0
    
    # For each batch and depth slice
    for b in range(B):
        for d_idx in depth_indices:
            if count < len(all_axes):
                # Extract the slice
                if C == 1:
                    slice_data = tensor[b, 0, d_idx]
                else:
                    # For RGB, ensure proper channel order
                    slice_data = np.transpose(tensor[b, :, d_idx], (1, 2, 0))
                
                # Plot the slice
                ax = all_axes[count]
                if C == 1:
                    im = ax.imshow(slice_data, cmap='gray')
                else:
                    im = ax.imshow(slice_data)
                
                ax.set_title(f"Batch {b}, Depth {d_idx}")
                ax.axis('off')
                count += 1
    
    # Hide any unused axes
    for i in range(count, len(all_axes)):
        all_axes[i].axis('off')
    
    plt.tight_layout()
    return fig
