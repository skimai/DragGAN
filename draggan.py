import os
import sys
import time
from typing import List, Optional, Tuple
import math

import numpy as np
import PIL
import torch
import streamlit as st

stylegan2_dir = os.path.abspath("stylegan2")
sys.path.insert(0, stylegan2_dir)
import dnnlib
import legacy

import utils


@st.cache_resource
def load_model(
    network_pkl: str = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl",
    device: torch.device = torch.device("cuda"),
    fp16: bool = True,
) -> torch.nn.Module:
    """
    Loads a pretrained StyleGAN2-ADA generator network from a pickle file.

    Args:
        network_pkl (str): The URL or local path to the network pickle file.
        device (torch.device): The device to use for the computation.
        fp16 (bool): Whether to use half-precision floating point format for the network weights.

    Returns:
        The pretrained generator network.
    """
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        chkpt = legacy.load_network_pkl(f, force_fp16=fp16)
    G = chkpt["G_ema"].to(device).eval()
    for param in G.parameters():
        param.requires_grad_(False)

    # Create a new attribute called "activations" for the Generator class
    # This will be a list of activations from each layer
    G.__setattr__("activations", None)

    # Forward hook to collect features
    def hook(module, input, output):
        G.activations = output

    # Apply the hook to the 7th layer (256x256)
    for i, (name, module) in enumerate(G.synthesis.named_children()):
        if i == 6:
            print("Registering hook for:", name)
            module.register_forward_hook(hook)

    return G


@st.cache_data()
def generate_W(
    _G: torch.nn.Module,
    seed: int = 0,
    network_pkl: Optional[str] = None,
    truncation_psi: float = 1.0,
    truncation_cutoff: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """
    Generates a latent code tensor in W+ space from a pretrained StyleGAN2-ADA generator network.

    Args:
        _G (torch.nn.Module): The generator network, with underscore to avoid streamlit cache error
        seed (int): The random seed to use for generating the latent code.
        network_pkl (Optional[str]): The path to the network pickle file. If None, the default network will be used.
        truncation_psi (float): The truncation psi value to use for the mapping network.
        truncation_cutoff (Optional[int]): The number of layers to use for the truncation trick. If None, all layers will be used.
        device (torch.device): The device to use for the computation.

    Returns:
        The W+ latent as a numpy array of shape [1, num_layers, 512].
    """
    G = _G
    torch.manual_seed(seed)
    z = torch.randn(1, G.z_dim).to(device)
    num_layers = G.synthesis.num_ws
    if truncation_cutoff == -1:
        truncation_cutoff = None
    elif truncation_cutoff is not None:
        truncation_cutoff = min(num_layers, truncation_cutoff)
    W = G.mapping(
        z,
        None,
        truncation_psi=truncation_psi,
        truncation_cutoff=truncation_cutoff,
    )
    return W.cpu().numpy()


def forward_G(
    G: torch.nn.Module,
    W: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass through the generator network.

    Args:
        G (torch.nn.Module): The generator network.
        W (torch.Tensor): The latent code tensor of shape [batch_size, latent_dim, 512].
        device (torch.device): The device to use for the computation.

    Returns:
        A tuple containing the generated image tensor of shape [batch_size, 3, height, width]
        and the feature maps tensor of shape [batch_size, num_channels, height, width].
    """
    if not isinstance(W, torch.Tensor):
        W = torch.from_numpy(W).to(device)

    img = G.synthesis(W, noise_mode="const", force_fp32=True)

    return img, G.activations[0]


@st.cache_data()
def generate_image(
    W,
    _G: Optional[torch.nn.Module] = None,
    network_pkl: Optional[str] = None,
    class_idx=None,
    device=torch.device("cuda"),
) -> Tuple[PIL.Image.Image, torch.Tensor]:
    """
    Generates an image using a pretrained generator network.

    Args:
        W (torch.Tensor): A tensor of latent codes of shape [batch_size, latent_dim, 512].
        _G (Optional[torch.nn.Module]): The generator network. If None, the network will be loaded from `network_pkl`.
        network_pkl (Optional[str]): The path to the network pickle file. If None, the default network will be used.
        class_idx (Optional[int]): The class index to use for conditional generation. If None, unconditional generation will be used.
        device (str): The device to use for the computation.

    Returns:
        A tuple containing the generated image as a PIL Image object and the feature maps tensor of shape [batch_size, num_channels, height, width].
    """
    if _G is None:
        assert network_pkl is not None
        _G = load_model(network_pkl, device)
    G = _G

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise Exception(
                "Must specify class label with --class when using a conditional network"
            )
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print("warn: --class=lbl ignored when running on an unconditional network")

    ## Generate image
    img, features = forward_G(G, W, device)

    img = utils.tensor_to_PIL(img)

    return img, features


def optimize(
    W: np.ndarray,
    G: torch.nn.Module,
    handle_points: List[Tuple[int, int]],
    target_points: List[Tuple[int, int]],
    r1: int = 3,
    r2: int = 12,
    tolerance: int = 2,
    max_iter: int = 200,
    lr: float = 0.1,
    multiplier: float = 1.0,
    lambda_: float = 0.1,
    device: torch.device = torch.device("cuda"),
    empty=None,
    display_every: int = 10,
    target_resolution: int = 512,
) -> np.ndarray:
    """
    Optimizes the latent code tensor W to generate an image that matches the target points.

    Args:
        W (np.ndarray): The initial latent code tensor of shape [1, num_layers, 512].
        G (torch.nn.Module): The generator network.
        handle_points (List[Tuple[int, int]]): The initial handle points as a list of (x, y) tuples.
        target_points (List[Tuple[int, int]]): The target points as a list of (x, y) tuples.
        r1 (int): The radius of the motion supervision loss.
        r2 (int): The radius of the point tracking.
        d (int): The tolerance for the handle points to reach the target points.
        max_iter (int): The maximum number of optimization iterations.
        lr (float): The learning rate for the optimizer.
        multiplier (float): The speed multiplier for the motion supervision loss.
        lambda_ (float): The weight of the motion supervision loss.
        device (torch.device): The device to use for the computation.
        empty: The st.empty object to display the intermediate images.
        display_every (int): The number of iterations between displaying intermediate images.
        target_resolution (int): The target resolution for the generated image.

    Returns:
        The optimized latent code tensor W as a numpy array of shape [1, num_layers, 512].
    """
    img, F0 = forward_G(G, W, device)
    empty.image(
        utils.tensor_to_PIL(img), caption="Initial image", width=target_resolution
    )

    F0_resized = torch.nn.functional.interpolate(
        F0,
        size=(target_resolution, target_resolution),
        mode="bilinear",
        align_corners=True,
    ).detach()

    # Convert handle/target points to tensors and reorder to [y, x]
    handle_points: torch.tensor = (
        torch.tensor(handle_points, device=device).flip(-1).float()
    )
    handle_points_0 = handle_points.clone()
    target_points: torch.tensor = (
        torch.tensor(target_points, device=device).flip(-1).float()
    )

    W = torch.from_numpy(W).to(device).float()
    W.requires_grad_(False)

    # Only optimize the first 6 layers of W
    W_layers_to_optimize = W[:, :6].clone()
    W_layers_to_optimize.requires_grad_(True)

    optimizer = torch.optim.Adam([W_layers_to_optimize], lr=lr)

    for i in range(max_iter):
        start = time.perf_counter()

        # # Check if the handle points have reached the target points
        if torch.allclose(handle_points, target_points, atol=tolerance):
            break

        optimizer.zero_grad()

        # Detach only the unoptimized layers
        W_combined = torch.cat([W_layers_to_optimize, W[:, 6:].detach()], dim=1)

        # Run the generator to get the image and feature maps
        img, F = forward_G(G, W_combined, device)

        ## Bilinear interpolate F to be same size as img
        F_resized = torch.nn.functional.interpolate(
            F,
            size=(target_resolution, target_resolution),
            mode="bilinear",
            align_corners=True,
        )

        # Compute the motion supervision loss
        loss, all_shifted_coordinates = motion_supervision(
            F_resized,
            F0_resized,
            handle_points,
            target_points,
            r1,
            lambda_,
            device,
            multiplier=multiplier,
        )

        # Backpropagate the loss and update the latent code
        loss.backward()

        # # Clip gradients if their norm exceeds max_grad_norm
        # torch.nn.utils.clip_grad_norm_(W_layers_to_optimize, 1.0)

        # # Compute the L2 regularization term
        # l2_regularization = 100 * torch.norm(W_layers_to_optimize - W[:, :6]) ** 2
        # print(l2_regularization.item())
        # # Add the regularization term to the loss
        # loss += l2_regularization

        optimizer.step()

        print(
            f"{i}\tLoss: {loss.item():0.2f}\tTime: {(time.perf_counter() - start) * 1000:.0f}ms"
        )

        if i % display_every == 0 or i == max_iter - 1:
            # Draw d_i intermediate target as orange ellipse
            img = utils.tensor_to_PIL(img)
            if img.size[0] != target_resolution:
                img = img.resize((target_resolution, target_resolution))
            draw = PIL.ImageDraw.Draw(img)

            # # Draw shifted coordinates handle + d_i
            # for points in all_shifted_coordinates:
            #     if not torch.isnan(points).any():
            #         coords = utils.get_ellipse_coords(points.mean(0).flip(-1).cpu().long().numpy().tolist(), 7)
            #         draw.ellipse(coords, fill="orange")

            for handle_point, target_point in zip(handle_points.flip(-1).cpu().long().numpy().tolist(), target_points.flip(-1).cpu().long().numpy().tolist()):
                # Draw the handle point
                handle_coords = utils.get_ellipse_coords(handle_point, 5)
                draw.ellipse(handle_coords, fill="red")

                # Draw the target point
                target_coords = utils.get_ellipse_coords(target_point, 5)
                draw.ellipse(target_coords, fill="blue")

                # Draw arrow head
                arrow_head_length = 10.0

                # Compute the direction vector of the line
                dx = target_point[0] - handle_point[0]
                dy = target_point[1] - handle_point[1]
                angle = math.atan2(dy, dx)

                # Shorten the target point by the length of the arrowhead
                shortened_target_point = (
                    target_point[0] - arrow_head_length * math.cos(angle),
                    target_point[1] - arrow_head_length * math.sin(angle),
                )
                
                # Draw the arrow (main line)
                draw.line([tuple(handle_point), shortened_target_point], fill='green', width=2)

                # Compute the points for the arrowhead
                arrow_point1 = (
                    target_point[0] - arrow_head_length * math.cos(angle - math.pi / 6),
                    target_point[1] - arrow_head_length * math.sin(angle - math.pi / 6),
                )

                arrow_point2 = (
                    target_point[0] - arrow_head_length * math.cos(angle + math.pi / 6),
                    target_point[1] - arrow_head_length * math.sin(angle + math.pi / 6),
                )

                # Draw the arrowhead
                draw.polygon([tuple(target_point), arrow_point1, arrow_point2], fill='green')

            empty.image(
                img, caption=f"iter: {i}, loss: {loss:.2f}", width=target_resolution
            )

        # Update the handle points with point tracking
        handle_points = point_tracking(
            F_resized,
            F0_resized,
            handle_points,
            handle_points_0,
            r2,
            device,
        )

    return torch.cat([W_layers_to_optimize, W[:, 6:]], dim=1).detach().cpu().numpy()


def motion_supervision(
    F: torch.Tensor,
    F0: torch.Tensor,
    handle_points: torch.Tensor,
    target_points: torch.Tensor,
    r1: int = 3,
    lambda_: float = 20.0,
    device: torch.device = torch.device("cuda"),
    multiplier: float = 1.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Computes the motion supervision loss and the shifted coordinates for each handle point.

    Args:
        F (torch.Tensor): The feature map tensor of shape [batch_size, num_channels, height, width].
        F0 (torch.Tensor): The original feature map tensor of shape [batch_size, num_channels, height, width].
        handle_points (torch.Tensor): The handle points tensor of shape [num_handle_points, 2].
        target_points (torch.Tensor): The target points tensor of shape [num_handle_points, 2].
        r1 (int): The radius of the circular mask around each handle point.
        lambda_ (float): The weight of the reconstruction loss for the unmasked region.
        device (torch.device): The device to use for the computation.
        multiplier (float): The multiplier to use for the direction vector.

    Returns:
        A tuple containing the motion supervision loss tensor and a list of shifted coordinates
        for each handle point, where each element in the list is a tensor of shape [num_points, 2].
    """
    n = handle_points.shape[0]  # Number of handle points
    loss = 0.0
    all_shifted_coordinates = []  # List of shifted patches

    for i in range(n):
        # Compute direction vector
        target2handle = target_points[i] - handle_points[i]
        d_i = target2handle / (torch.norm(target2handle) + 1e-7) * multiplier
        if torch.norm(d_i) > torch.norm(target2handle):
            d_i = target2handle

        # Compute the mask for the pixels within radius r1 of the handle point
        mask = utils.create_circular_mask(
            F.shape[2], F.shape[3], center=handle_points[i].tolist(), radius=r1
        ).to(device)
        # mask = utils.create_square_mask(F.shape[2], F.shape[3], center=handle_points[i].tolist(), radius=r1).to(device)

        # Find indices where mask is True
        coordinates = torch.nonzero(mask).float()  # shape [num_points, 2]

        # Shift the coordinates in the direction d_i
        shifted_coordinates = coordinates + d_i[None]
        all_shifted_coordinates.append(shifted_coordinates)

        h, w = F.shape[2], F.shape[3]

        # Extract features in the mask region and compute the loss
        F_qi = F[:, :, mask]  # shape: [C, H*W]

        # Sample shifted patch from F
        normalized_shifted_coordinates = shifted_coordinates.clone()
        normalized_shifted_coordinates[:, 0] = (
            2.0 * shifted_coordinates[:, 0] / (h - 1)
        ) - 1  # for height
        normalized_shifted_coordinates[:, 1] = (
            2.0 * shifted_coordinates[:, 1] / (w - 1)
        ) - 1  # for width
        # Add extra dimensions for batch and channels (required by grid_sample)
        normalized_shifted_coordinates = normalized_shifted_coordinates.unsqueeze(
            0
        ).unsqueeze(
            0
        )  # shape [1, 1, num_points, 2]
        normalized_shifted_coordinates = normalized_shifted_coordinates.flip(
            -1
        )  # grid_sample expects [x, y] instead of [y, x]
        normalized_shifted_coordinates = normalized_shifted_coordinates.clamp(-1, 1)

        # Use grid_sample to interpolate the feature map F at the shifted patch coordinates
        F_qi_plus_di = torch.nn.functional.grid_sample(
            F, normalized_shifted_coordinates, mode="bilinear", align_corners=True
        )
        # Output has shape [1, C, 1, num_points] so squeeze it
        F_qi_plus_di = F_qi_plus_di.squeeze(2)  # shape [1, C, num_points]

        loss += torch.nn.functional.l1_loss(F_qi.detach(), F_qi_plus_di)

    # TODO: add reconstruction loss for the unmasked region
    # # Add reconstruction loss for the unmasked region
    # loss += lambda_ * torch.norm((F - F0) * (1 - mask_total), p=1)

    return loss, all_shifted_coordinates


def point_tracking(
    F: torch.Tensor,
    F0: torch.Tensor,
    handle_points: torch.Tensor,  # [N, y, x]
    handle_points_0: torch.Tensor,  # [N, y, x]
    r2: int = 3,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Tracks the movement of handle points in an image using feature matching.

    Args:
        F (torch.Tensor): The feature maps tensor of shape [batch_size, num_channels, height, width].
        F0 (torch.Tensor): The feature maps tensor of shape [batch_size, num_channels, height, width] for the initial image.
        handle_points (torch.Tensor): The handle points tensor of shape [N, y, x].
        handle_points_0 (torch.Tensor): The handle points tensor of shape [N, y, x] for the initial image.
        r2 (int): The radius of the patch around each handle point to use for feature matching.
        device (torch.device): The device to use for the computation.

    Returns:
        The new handle points tensor of shape [N, y, x].
    """
    n = handle_points.shape[0]  # Number of handle points
    new_handle_points = torch.zeros_like(handle_points)

    for i in range(n):
        # Compute the patch around the handle point
        patch = utils.create_square_mask(
            F.shape[2], F.shape[3], center=handle_points[i].tolist(), radius=r2
        ).to(device)

        # Find indices where the patch is True
        patch_coordinates = torch.nonzero(patch)  # shape [num_points, 2]

        # Extract features in the patch 
        F_qi = F[
            :, :, patch_coordinates[:, 0], patch_coordinates[:, 1]
        ]
        # Extract feature of the initial handle point
        f_i = F0[
            :, :, handle_points_0[i][0].long(), handle_points_0[i][1].long()
        ]

        # Compute the L1 distance between the patch features and the initial handle point feature
        distances = torch.norm(F_qi - f_i[:, :, None], p=1, dim=1)

        # Find the new handle point as the one with minimum distance
        min_index = torch.argmin(distances)
        new_handle_points[i] = patch_coordinates[min_index]

    return new_handle_points
