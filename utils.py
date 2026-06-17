import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.hub import load
import torch.nn.functional as F

class DepthEstimator:
    def __init__(self, model_type='MiDaS_small', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load("intel-isl/MiDaS", model_type).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_batch(self, images):
        # images = torch.stack(images)
        return self.transform(images).to(self.device)

    def estimate_depth(self, images):
        with torch.no_grad():
            original_size = images.shape[-2:]  # Assume all images have the same size
            input_batch = self.preprocess_batch(images)
            depth_maps = self.model(input_batch)
            
            # Resize depth maps back to original dimensions
            depth_maps_resized = F.interpolate(
                depth_maps.unsqueeze(1), size=original_size, mode="bicubic", align_corners=False
            ).squeeze(1)
            
            # Normalize depth maps to range [0, 1]
            depth_min = depth_maps_resized.amin(dim=(1, 2), keepdim=True)
            depth_max = depth_maps_resized.amax(dim=(1, 2), keepdim=True)
            depth_normalized = (depth_maps_resized - depth_min) / (depth_max - depth_min)
        return depth_normalized.unsqueeze(1).permute(0, 2, 3, 1)


# import torch
# import torch.nn.functional as F
# from typing import Optional

@torch.jit.script
def draw_random_vertical_lines(
    depth_tensor: torch.Tensor,
    num_lines: int,
    min_thickness: int,
    max_thickness: int,
    line_value: float
) -> torch.Tensor:
    """
    Draws `num_lines` random vertical lines on a single-channel depth image (H, W),
    with random line thickness between [min_thickness, max_thickness].

    Modifies the tensor in-place.

    Args:
        depth_tensor (torch.Tensor): Single depth image (H, W).
        num_lines (int): How many random vertical lines to draw.
        min_thickness (int): Minimum line thickness in pixels.
        max_thickness (int): Maximum line thickness in pixels.
        line_value (float): Pixel value for the lines (e.g., "white" depth).
    
    Returns:
        torch.Tensor: Modified depth image with vertical lines.
    """
    H = depth_tensor.shape[0]
    W = depth_tensor.shape[1]

    for _i in range(num_lines):
        # Random x-position
        x_rand = torch.randint(low=0, high=W, size=(1,))
        x = int(x_rand[0].item())

        # Random thickness
        thickness_rand = torch.randint(low=min_thickness, high=max_thickness + 1, size=(1,))
        line_thickness = int(thickness_rand[0].item())

        # Random y start and end
        y1_rand = torch.randint(low=0, high=H, size=(1,))
        y1 = int(y1_rand[0].item())
        y2_rand = torch.randint(low=0, high=H, size=(1,))
        y2 = int(y2_rand[0].item())

        # Ensure y1 <= y2
        if y2 < y1:
            temp = y1
            y1 = y2
            y2 = temp

        # Horizontal bounds for thickness
        x_min = max(0, x - line_thickness // 2)
        x_max = min(W, x + line_thickness // 2 + 1)

        # Vertical clamp
        y1_clamped = max(0, y1)
        y2_clamped = min(H - 1, y2)

        if y2_clamped >= y1_clamped:
            depth_tensor[y1_clamped : y2_clamped + 1, x_min : x_max] = line_value

    return depth_tensor


@torch.jit.script
def add_depth_noise(
    depth_image: torch.Tensor,
    gaussian_std: float = 0.05,
    alpha: float = 0.001,
    beta: float = 0.002,
    gamma: float = 0.001,
    drop_prob: float = 0.05,
    speckle_var: float = 0.01,
    edge_drop_prob: float = 0.9,
    blur_kernel: int = 5,
    edge_dilation: int = 3,
    blur_iterations: int = 2,
    add_white_lines: bool = True,
    max_lines_per_image: int = 5,
    min_line_thickness: int = 1,
    max_line_thickness: int = 7
) -> torch.Tensor:
    """
    TorchScript-compatible pipeline that adds:
      - Gaussian noise
      - Depth-dependent noise
      - Edge-based dropout
      - Speckle noise
      - Optional random vertical white lines (with varying thickness)
      - Gaussian blur

    Args:
        depth_image (torch.Tensor): (N, 1, H, W) depth images.
        gaussian_std (float): Std dev of Gaussian noise.
        alpha, beta, gamma (float): Coeffs for depth-dependent noise.
        drop_prob (float): Probability of dropout (NaN).
        speckle_var (float): Variance for speckle (multiplicative) noise.
        edge_drop_prob (float): Probability for edge dropout.
        blur_kernel (int): Kernel size for Gaussian blur (must be odd).
        edge_dilation (int): Kernel size for dilating edges.
        blur_iterations (int): Number of consecutive blur passes.
        add_white_lines (bool): Whether to add vertical white lines.
        max_lines_per_image (int): Max number of white lines per image.
        min_line_thickness (int): Minimum thickness of each line.
        max_line_thickness (int): Maximum thickness of each line.

    Returns:
        torch.Tensor: Noisy depth images (N, 1, H, W).
    """
    shape = depth_image.shape
    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]
    if C != 1:
        raise RuntimeError("depth_image must have shape (N, 1, H, W).")

    # 1. Gaussian noise
    noise_gaussian = torch.randn_like(depth_image) * gaussian_std
    depth_noisy = depth_image + noise_gaussian

    # 2. Depth-dependent noise (alpha*z^2 + beta*z + gamma)
    std_depth = alpha * depth_image * depth_image + beta * depth_image + gamma
    noise_depth = torch.randn_like(depth_image) * std_depth
    depth_noisy = depth_noisy + noise_depth

    # 3. Edge detection (Sobel on original depth_image)
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=depth_image.device).reshape(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)

    grad_x = F.conv2d(depth_image, sobel_x, padding=1)  # (N,1,H,W)
    grad_y = F.conv2d(depth_image, sobel_y, padding=1)  # (N,1,H,W)
    grad_mag = torch.sqrt(grad_x*grad_x + grad_y*grad_y)

    # Normalize gradient magnitude per image
    grad_flat = grad_mag.view(N, -1)
    max_vals, _ = torch.max(grad_flat, dim=1, keepdim=True)
    max_vals = max_vals.unsqueeze(-1)  # shape => (N,1,1)
    eps = 1e-6
    edges_norm = grad_mag / (max_vals.view(N, 1, 1, 1) + eps)  # [0..1]

    # 4. Edge dilation
    if edge_dilation > 1:
        edges_norm = F.max_pool2d(edges_norm, kernel_size=edge_dilation, stride=1,
                                  padding=edge_dilation // 2)

    # 5. Missing data (edge-based + random dropout)
    prob_map = edges_norm * edge_drop_prob
    rnd_for_edges = torch.rand_like(prob_map)
    keep_edges = (rnd_for_edges > prob_map)

    rnd_for_dropout = torch.rand_like(depth_noisy)
    keep_rand = (rnd_for_dropout > drop_prob)

    keep_mask = keep_edges & keep_rand
    depth_noisy = torch.where(keep_mask, depth_noisy, torch.tensor(float('nan'), device=depth_noisy.device))

    # 6. Speckle noise (multiplicative)
    noise_speckle = torch.randn_like(depth_noisy) * speckle_var * depth_noisy
    depth_noisy = depth_noisy + noise_speckle

    # 7. Fill NaNs with local average
    nan_mask = torch.isnan(depth_noisy)
    if torch.any(nan_mask):
        zero_filled = torch.where(nan_mask, torch.zeros_like(depth_noisy), depth_noisy)
        depth_filled = F.avg_pool2d(zero_filled, kernel_size=3, stride=1, padding=1)
        depth_noisy = torch.where(nan_mask, depth_filled, depth_noisy)

    # 8. Gaussian blur
    if blur_kernel > 1 and (blur_kernel % 2) == 1:
        sigma = float(blur_kernel) / 6.0
        coords = torch.arange(blur_kernel, device=depth_image.device) - (blur_kernel // 2)
        coords_x = coords.unsqueeze(0)
        coords_y = coords.unsqueeze(1)
        kernel_2d = torch.exp(-(coords_x*coords_x + coords_y*coords_y) / (2.0*sigma*sigma))
        kernel_2d = kernel_2d / torch.sum(kernel_2d)
        gauss_kernel = kernel_2d.view(1,1,blur_kernel,blur_kernel)

        for _j in range(blur_iterations):
            depth_noisy = F.conv2d(depth_noisy, gauss_kernel, padding=blur_kernel // 2)

    # 9. (Optional) Add random vertical lines with varying thickness
    if add_white_lines and max_lines_per_image > 0:
        for i in range(N):
            # For each image in the batch, pick a local max
            image_2d = depth_noisy[i, 0, :, :]
            local_max = float(torch.max(image_2d))
            line_value = local_max * 1.1 if local_max > 0.0 else 1.0

            # Random number of lines in [0, max_lines_per_image]
            num_lines_rand = torch.randint(low=0, high=max_lines_per_image+1, size=(1,))
            num_lines = int(num_lines_rand[0].item())

            draw_random_vertical_lines(
                image_2d,
                num_lines=num_lines,
                min_thickness=min_line_thickness,
                max_thickness=max_line_thickness,
                line_value=line_value
            )

    return depth_noisy
