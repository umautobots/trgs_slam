from typing import Literal, Tuple, Dict, Union, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import cv2
import torch

import lietorch as lt
from gsplat.rendering import rasterization

from trgs_slam.gaussians.gaussians import GaussianManager
from trgs_slam.renderer.fpn import FPNConfig, FPNManager
from trgs_slam.trajectory.trajectory import TrajectoryManager, to_transformation_matrices
from trgs_slam.tools.nested_timer import NestedTimer
from trgs_slam.datasets.base_dataset import BaseDataset

@dataclass
class RendererConfig:
    # The number of sharp images to rasterize for each blurry/rolling shutter image rendered. Set to 1 to disable
    # modeling of blur and rolling shutter.
    num_rasters: int = 5
    # The maximum integration interval used for modeling motion blur. Ignored if num_rasters == 1.
    integration_interval: float = 36e-3
    # Rasterization mode: 'classic' (from the original 3DGS paper) or 'antialiased' (from "Mip-Splatting").
    rasterize_mode: Literal['classic', 'antialiased'] = 'antialiased'
    # Whether to apply a random background to rendered images to discourage transparency when mapping.
    use_random_background: bool = True
    # FPN options.
    fpn_config: FPNConfig = field(default_factory=FPNConfig)

class Renderer:
    def __init__(
        self,
        config: RendererConfig,
        dataset: BaseDataset,
        gaussian_manager: GaussianManager,
        trajectory_device: str,
        time_start: float,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.gaussian_manager = gaussian_manager
        self.gaussian_device = self.gaussian_manager.config.device
        self.trajectory_device = trajectory_device
        self.new_camera_matrix = torch.from_numpy(self.dataset.new_camera_matrix).to(
            dtype=torch.float, device=self.gaussian_device)

        # Initialize the FPN manager.
        self.fpn_manager = FPNManager(self.config.fpn_config, self.dataset, time_start, self.gaussian_device)

        # Compute the time offsets and the integration weights.
        assert self.config.num_rasters > 0, 'num_rasters must be positive.'
        if self.config.num_rasters > 1:
            self.compute_offsets_and_weights()

    def compute_offsets_and_weights(
        self,
    ) -> None:
        # Compute the rolling shutter lookup table.
        x_coords, y_coords = np.meshgrid(np.arange(self.dataset.image_width), np.arange(self.dataset.image_height))
        readout_pixel_duration = self.dataset.readout_row_duration / self.dataset.image_width
        t_roll_lookup_distorted = \
            x_coords * readout_pixel_duration + \
            y_coords * self.dataset.readout_row_duration
        t_roll_lookup_undistorted = torch.from_numpy(
            cv2.remap(t_roll_lookup_distorted, self.dataset.mapx, self.dataset.mapy, cv2.INTER_LINEAR)).to(
            dtype=torch.double, device=self.gaussian_device)

        # Compute the time offsets for the renders.
        self.time_offsets = torch.linspace(
            t_roll_lookup_undistorted.max() - self.config.integration_interval,
            t_roll_lookup_undistorted.max(),
            self.config.num_rasters, device=self.gaussian_device, dtype=torch.double)

        # Compute the pixelwise brightness correction.
        brightness_correction = (1 /
            (1 - torch.exp((self.time_offsets[0] - t_roll_lookup_undistorted) / self.dataset.thermal_time_constant)))

        # Compute the integration weights
        integration_weights = torch.zeros(
            self.config.num_rasters, self.dataset.image_height, self.dataset.image_width,
            dtype=torch.float, device=self.gaussian_device)
        delta_t = self.config.integration_interval / (self.config.num_rasters - 1)
        js = torch.floor((t_roll_lookup_undistorted - self.time_offsets[0]) / delta_t).int()
        for i, time_offset in enumerate(self.time_offsets):
            # Compute the time offsets between the readout time and render times.
            time_diff = t_roll_lookup_undistorted - time_offset
            time_diff_prev = time_diff + delta_t
            time_diff_next = time_diff - delta_t

            # Compute the exponential factors based on the thermal time constant.
            exp_factor = torch.exp(-time_diff / self.dataset.thermal_time_constant)
            exp_factor_prev = torch.exp(-time_diff_prev / self.dataset.thermal_time_constant)
            exp_factor_next = torch.exp(-time_diff_next / self.dataset.thermal_time_constant)

            # Compute the ratio of the thermal time constant to the render time delta.
            ratio_tau_dt = self.dataset.thermal_time_constant / delta_t

            # Determine pixels that are readout between the current render time and the next render time.
            j_zero = js == 0

            # Add the contribution from the partial segment to the render weight for the j == 0 pixels.
            integration_weights[i][j_zero] += 1 - time_diff[j_zero] / delta_t + ratio_tau_dt - \
                exp_factor[j_zero] * (1 + ratio_tau_dt)

            # Determine pixels that are readout between the previous render time and the current render time.
            j_m_one = js == -1

            # Add the contribution from the partial segment to the render weight for the j == -1 pixels.
            integration_weights[i][j_m_one] += time_diff_prev[j_m_one] / delta_t - ratio_tau_dt + \
                exp_factor_prev[j_m_one] * ratio_tau_dt

            # Determine the pixels that are readout after the next render time.
            j_pos = js > 0

            if i != 0:
                # Add the contribution from the full segment to the render weight for the j == 0 pixels.
                integration_weights[i][j_zero] += exp_factor[j_zero] * (1 - ratio_tau_dt) + \
                    exp_factor_prev[j_zero] * ratio_tau_dt

                # Add the contribution from the full segments to the render weight for the j > 0 pixels.
                integration_weights[i][j_pos] += ratio_tau_dt * \
                    (exp_factor_prev[j_pos] - 2 * exp_factor[j_pos] + exp_factor_next[j_pos])
            else:
                # Add the contribution from the full segment to the render weight for the j > 0 pixels.
                integration_weights[i][j_pos] += exp_factor_next[j_pos] * ratio_tau_dt - \
                    exp_factor[j_pos] * (1 + ratio_tau_dt)

            # Decrement js.
            js -= 1

        # Apply the pixelwise brightness correction to the integration weights.
        self.integration_weights = integration_weights * brightness_correction.unsqueeze(0).float()

        # Move the time offsets to the trajectory device.
        self.time_offsets = self.time_offsets.to(self.trajectory_device)

    @NestedTimer.timed_function('Rendering')
    def render(
        self,
        render_timestamps: torch.Tensor,
        trajectory_manager: TrajectoryManager,
        model_motion_artifacts: bool,
        mode: Literal['mapping', 'tracking'] = 'mapping',
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], torch.Tensor, lt.SO3]:
        # Set the raster timestamps.
        num_renders = len(render_timestamps)
        if model_motion_artifacts and self.config.num_rasters > 1:
            raster_timestamps = \
                render_timestamps.repeat_interleave(self.config.num_rasters) + self.time_offsets.repeat(num_renders)
        else:
            raster_timestamps = render_timestamps

        with NestedTimer('Pose Evaluation'):
            # Evaluate the camera poses at the raster timestamps.
            positions_camera_in_world, rotations_camera_to_world = \
                trajectory_manager(raster_timestamps, as_matrices=False)
            transformations_world_to_camera = \
                to_transformation_matrices(positions_camera_in_world, rotations_camera_to_world, invert=True).to(
                    dtype=torch.float, device=self.gaussian_device)

        with NestedTimer('Rasterization'):
            # Rasterize images at the camera poses.
            rasterized_images, rasterized_alphas, info = self.rasterize(transformations_world_to_camera)
            rasterized_images = rasterized_images.squeeze(-1)
            rasterized_alphas = rasterized_alphas.squeeze(-1)
            with torch.no_grad():
                info['directions'] = self.gaussian_manager.params['means'][None, :, :] - \
                    positions_camera_in_world[:, None, :].to(dtype=torch.float, device=self.gaussian_device)

        with NestedTimer('Raster Post Processing'):
            # Blend the rasterized images with a random background.
            if self.config.use_random_background and mode == 'mapping':
                random_intensity = torch.rand(1, device=self.gaussian_device)
                rasterized_images = \
                    rasterized_images + random_intensity * (1.0 - rasterized_alphas)

            # Compute the blurry and rolling shutter renders.
            if model_motion_artifacts and self.config.num_rasters > 1:
                rendered_images_clean = \
                    (rasterized_images * self.integration_weights.repeat(num_renders, 1, 1)).reshape(
                    num_renders, self.config.num_rasters, self.dataset.image_height, self.dataset.image_width).sum(
                        dim=1)
                info['integration_weights'] = self.integration_weights
            else:
                rendered_images_clean = rasterized_images

            # Apply the FPN offsets.
            rendered_images = self.fpn_manager(rendered_images_clean, render_timestamps)

        output_images = {
            'rasterized_images': rasterized_images,
            'rendered_images_clean': rendered_images_clean,
            'rendered_images': rendered_images}

        return output_images, info, positions_camera_in_world, rotations_camera_to_world

    def rasterize(
        self,
        transformations_world_to_camera: torch.Tensor,
        intensities: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        num_rasters = len(transformations_world_to_camera)
        rasterized_images, rasterized_alphas, info = rasterization(
            means=self.gaussian_manager.params['means'],
            quats=self.gaussian_manager.params['quats'],
            scales=torch.exp(self.gaussian_manager.params['scales']),
            opacities=torch.sigmoid(self.gaussian_manager.params['opacities']),
            colors=torch.sigmoid(self.gaussian_manager.params['intensities']) if intensities is None else intensities,
            viewmats=transformations_world_to_camera,
            Ks=kwargs.pop('Ks', self.new_camera_matrix.unsqueeze(0).repeat(num_rasters, 1, 1)),
            width=kwargs.pop('width', self.dataset.image_width),
            height=kwargs.pop('height', self.dataset.image_height),
            packed=False,
            absgrad=self.gaussian_manager.config.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
            **kwargs)

        return rasterized_images, rasterized_alphas, info

    def zero_grad(
        self
    ) -> None:
        self.fpn_manager.zero_grad()

    def get_loss(
        self,
        times_eval: torch.Tensor,
    ) -> Union[torch.Tensor, float]:
        return self.fpn_manager.get_loss(times_eval)

    def step(
        self,
    ) -> None:
        self.fpn_manager.step()

    def freeze(
        self,
    ) -> None:
        self.fpn_manager.freeze()

    def unfreeze(
        self,
    ) -> None:
        self.fpn_manager.unfreeze()

    def get_copy(
        self,
    ) -> Dict[str, Any]:
        return self.fpn_manager.get_copy()

    def save(
        self,
        base_result_dir: str,
        suffix: Optional[str] = None,
        save_optimizer_states: bool = True,
        save_freeze_states: bool = True,
    ) -> None:
        result_dir = f'{base_result_dir}/renderer'
        self.fpn_manager.save(result_dir, suffix, save_optimizer_states, save_freeze_states)

    def load_copy(
        self,
        save_data: Dict[str, Any],
    ) -> None:
        self.fpn_manager.load_copy(save_data)

    def load(
        self,
        base_result_dir: str,
        suffix: Optional[str] = None,
        load_optimizer_states: bool = True,
        load_freeze_states: bool = True,
    ) -> None:
        result_dir = f'{base_result_dir}/renderer'
        self.fpn_manager.load(result_dir, suffix, load_optimizer_states, load_freeze_states)
