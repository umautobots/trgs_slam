from typing import Union, Optional, Dict, Any
from dataclasses import dataclass
import os
import copy

import torch

from lie_spline_torch.uniform_rd_bspline import UniformRdBSpline

from trgs_slam.datasets.base_dataset import BaseDataset

@dataclass
class FPNConfig:
    # Whether to enable FPN modeling.
    enable: bool = True
    # Order of the pixelwise offsets spline.
    pixelwise_spline_order: int = 2
    # Order of the scalar offsets spline.
    scalar_spline_order: int = 2
    # Knot time interval for the pixelwise offsets spline (in seconds).
    pixelwise_knot_interval: float = 30.0
    # Knot time interval for the scalar offsets spline (in seconds).
    scalar_knot_interval: float = 5.0
    # Learning rate for pixelwise offsets.
    pixelwise_lr: float = 1e-4
    # Learning rate for scalar offsets.
    scalar_lr: float = 1e-4
    # Weight for the pixelwise offsets loss.
    pixelwise_weight: float = 1e-3

class FPN(torch.nn.Module):
    def __init__(
        self,
        config: FPNConfig,
        image_height: int,
        image_width: int,
        time_start: float,
        pixelwise_control_points: Optional[torch.Tensor] = None,
        scalar_control_points: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.image_height = image_height
        self.image_width = image_width
        self.time_start = time_start

        if pixelwise_control_points is None:
            pixelwise_control_points = torch.zeros((self.config.pixelwise_spline_order, image_height * image_width),
                dtype=torch.float)
        self.pixelwise_offsets_spline = UniformRdBSpline(self.config.pixelwise_spline_order,
            self.config.pixelwise_knot_interval, time_start, pixelwise_control_points, sparse=True)

        if scalar_control_points is None:
            scalar_control_points = torch.zeros((self.config.scalar_spline_order, 1), dtype=torch.float)
        self.scalar_offsets_spline = UniformRdBSpline(self.config.scalar_spline_order, self.config.scalar_knot_interval,
            time_start, scalar_control_points, sparse=True)

    def forward(
        self,
        images: torch.Tensor,
        times_eval: torch.Tensor,
        add: bool = True,
    ) -> torch.Tensor:
        """Apply the FPN offsets to add or remove noise."""

        scalar_offsets = self.scalar_offsets_spline(times_eval).unsqueeze(-1)
        pixelwise_offsets = \
            self.pixelwise_offsets_spline(times_eval).reshape(len(times_eval), self.image_height, self.image_width)
        offsets = scalar_offsets + pixelwise_offsets

        return images + offsets if add else images - offsets

    def get_loss(
        self,
        times_eval: torch.Tensor,
    ) -> torch.Tensor:
        return self.pixelwise_offsets_spline(times_eval).mean().abs()

class FPNManager():
    def __init__(
        self,
        config: FPNConfig,
        dataset: BaseDataset,
        time_start: float,
        device: str,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.time_start = time_start
        self.latest_time_eval = time_start
        self.device = device

        if self.config.enable:
            self.fpn = FPN(self.config, self.dataset.image_height, self.dataset.image_width, time_start).to(self.device)
            self.pixelwise_offsets_optimizer = torch.optim.SparseAdam(
                self.fpn.pixelwise_offsets_spline.parameters(), lr=self.config.pixelwise_lr)
            self.scalar_offsets_optimizer = torch.optim.SparseAdam(
                self.fpn.scalar_offsets_spline.parameters(), lr=self.config.scalar_lr)

    def __call__(
        self,
        images: torch.Tensor,
        times_eval: torch.Tensor,
        add: bool = True,
    ) -> torch.Tensor:
        if not self.config.enable:
            return images

        if times_eval.max() > self.latest_time_eval:
            self.latest_time_eval = times_eval.max().item()
            self.fpn.pixelwise_offsets_spline.extend_to_time(self.latest_time_eval, self.pixelwise_offsets_optimizer)
            self.fpn.scalar_offsets_spline.extend_to_time(self.latest_time_eval, self.scalar_offsets_optimizer)

        return self.fpn(images, times_eval, add)

    def zero_grad(
        self,
    ) -> None:
        if self.config.enable:
            self.pixelwise_offsets_optimizer.zero_grad(set_to_none=True)
            self.scalar_offsets_optimizer.zero_grad(set_to_none=True)

    def get_loss(
        self,
        times_eval: torch.Tensor,
    ) -> Union[torch.Tensor, float]:
        if self.config.enable:
            return self.fpn.get_loss(times_eval) * self.config.pixelwise_weight
        else:
            return 0.0

    def step(
        self,
    ) -> None:
        if self.config.enable:
            self.pixelwise_offsets_optimizer.step()
            self.scalar_offsets_optimizer.step()

    def freeze(
        self,
    ) -> None:
        if self.config.enable:
            self.fpn.pixelwise_offsets_spline.control_points.requires_grad = False
            self.fpn.scalar_offsets_spline.control_points.requires_grad = False

    def unfreeze(
        self,
    ) -> None:
        if self.config.enable:
            self.fpn.pixelwise_offsets_spline.control_points.requires_grad = True
            self.fpn.scalar_offsets_spline.control_points.requires_grad = True

    def get_copy(
        self,
        save_optimizer_states: bool = True,
        save_freeze_states: bool = True,
    ) -> Dict[str, Any]:
        if not self.config.enable:
            return

        manager_state = {
            'pixelwise_spline_order': self.config.pixelwise_spline_order,
            'scalar_spline_order': self.config.scalar_spline_order,
            'pixelwise_knot_interval': self.config.pixelwise_knot_interval,
            'scalar_knot_interval': self.config.scalar_knot_interval,
            'image_height': self.dataset.image_height,
            'image_width': self.dataset.image_width,
            'time_start': self.time_start,
            'latest_time_eval': self.latest_time_eval}
        save_data = {
            'manager_state': manager_state,
            'parameter_states': copy.deepcopy(self.fpn.state_dict())}

        if save_optimizer_states:
            save_data['optimizer_states'] = {}
            save_data['optimizer_states']['pixelwise_offsets'] = \
                copy.deepcopy(self.pixelwise_offsets_optimizer.state_dict())
            save_data['optimizer_states']['scalar_offsets'] = copy.deepcopy(self.scalar_offsets_optimizer.state_dict())

        if save_freeze_states:
            save_data['freeze_states'] = {}
            save_data['freeze_states']['pixelwise_offsets'] = \
                self.fpn.pixelwise_offsets_spline.control_points.requires_grad
            save_data['freeze_states']['scalar_offsets'] = \
                self.fpn.scalar_offsets_spline.control_points.requires_grad

        return save_data

    def save(
        self,
        base_result_dir: str,
        suffix: Optional[str] = None,
        save_optimizer_states: bool = True,
        save_freeze_states: bool = True,
    ) -> None:
        if not self.config.enable:
            return

        result_dir = f'{base_result_dir}/fpn'
        if suffix is not None:
            filename = f'fpn_{suffix}.pt'
        else:
            filename = 'fpn.pt'
        os.makedirs(result_dir, exist_ok=True)
        filepath = os.path.join(result_dir, filename)

        torch.save(self.get_copy(save_optimizer_states, save_freeze_states), filepath)

    def load_copy(
        self,
        save_data: Dict[str, Any],
        load_optimizer_states: bool = True,
        load_freeze_states: bool = True,
    ) -> None:
        if not self.config.enable:
            return

        assert \
            save_data['manager_state']['pixelwise_spline_order'] == self.config.pixelwise_spline_order and \
            save_data['manager_state']['scalar_spline_order'] == self.config.scalar_spline_order and \
            save_data['manager_state']['pixelwise_knot_interval'] == self.config.pixelwise_knot_interval and \
            save_data['manager_state']['scalar_knot_interval'] == self.config.scalar_knot_interval and \
            save_data['manager_state']['time_start'] == self.time_start, \
            'The current and saved spline orders, knot intervals, and start time must match.'

        # Handle reshaping (upsampling/downsampling), if needed.
        old_height = save_data['manager_state']['image_height']
        old_width = save_data['manager_state']['image_width']
        new_height = self.dataset.image_height
        new_width = self.dataset.image_width
        if new_height == old_height and new_width == old_width:
            def reshape_fn(old_tensor):
                return old_tensor
        else:
            if new_height > old_height and new_width > old_width:
                # Upsample the pixelwise FPN control points and corresponding optimizer states.
                assert new_height % old_height == 0 and new_width % old_width == 0, \
                    f'The new FPN dimensions ({new_height}x{new_width}) must be an integer multiple of ' \
                    f'the saved FPN dimensions ({old_height}x{old_width}) for upsampling.'

                print('The current image resolution is greater than the stored image resolution. Upsampling FPN.')
                def reshape_fn(old_tensor):
                    return torch.nn.functional.interpolate(
                        old_tensor.reshape(old_tensor.shape[0], 1, old_height, old_width),
                        size=(new_height, new_width),
                        mode='nearest').reshape(old_tensor.shape[0], new_height * new_width)
            elif new_height < old_height and new_width < old_width:
                # Downsample the pixelwise FPN control points and corresponding optimizer states.
                assert old_height % new_height == 0 and old_width % new_width == 0, \
                    f'The old FPN dimensions ({old_height}x{old_width}) must be an integer multiple of ' \
                    f'the new FPN dimensions ({new_height}x{new_width}) for downsampling.'
                assert old_height // new_height == old_width // new_width, \
                    'The width and height downsample factors must match.'

                print('The current image resolution is less than the stored image resolution. Downsampling FPN.')
                downsample_factor = old_height // new_height
                def reshape_fn(old_tensor):
                    return torch.nn.functional.avg_pool2d(
                        old_tensor.reshape(old_tensor.shape[0], 1, old_height, old_width),
                        kernel_size=downsample_factor,
                        stride=downsample_factor).reshape(old_tensor.shape[0], new_height * new_width)
            else:
                raise ValueError(
                    f'The new FPN dimensions ({new_height}x{new_width}) are incompatible with ' \
                    f'the saved FPN dimensions ({old_height}x{old_width}).')

        self.latest_time_eval = save_data['manager_state']['latest_time_eval']
        pixelwise_control_points = reshape_fn(save_data['parameter_states']['pixelwise_offsets_spline.control_points'])
        self.fpn = FPN(self.config, new_height, new_width, self.time_start, pixelwise_control_points,
            save_data['parameter_states']['scalar_offsets_spline.control_points']).to(self.device)
        self.pixelwise_offsets_optimizer = torch.optim.SparseAdam(
            self.fpn.pixelwise_offsets_spline.parameters(), lr=self.config.pixelwise_lr)
        self.scalar_offsets_optimizer = torch.optim.SparseAdam(
            self.fpn.scalar_offsets_spline.parameters(), lr=self.config.scalar_lr)

        if load_optimizer_states:
            if 'optimizer_states' in save_data:
                self.scalar_offsets_optimizer.load_state_dict(save_data['optimizer_states']['scalar_offsets'])

                pixelwise_opt_state = save_data['optimizer_states']['pixelwise_offsets']
                param_state = pixelwise_opt_state['state'][0]
                for key in param_state.keys():
                    if isinstance(param_state[key], torch.Tensor) and param_state[key].dim() != 0:
                        param_state[key] = reshape_fn(param_state[key])
                self.pixelwise_offsets_optimizer.load_state_dict(pixelwise_opt_state)
            else:
                print('Warning: no FPN optimizer states were saved. The FPN optimizers have been initialized with '
                    'default states')

        if load_freeze_states:
            if 'freeze_states' in save_data:
                self.fpn.pixelwise_offsets_spline.control_points.requires_grad = \
                    save_data['freeze_states']['pixelwise_offsets']
                self.fpn.scalar_offsets_spline.control_points.requires_grad = \
                    save_data['freeze_states']['scalar_offsets']
            else:
                print('Warning: no freeze states were saved. The parameters have been initialized with '
                    'requires_grad == True')

    def load(
        self,
        base_result_dir: str,
        suffix: Optional[str] = None,
        load_optimizer_states: bool = True,
        load_freeze_states: bool = True,
    ) -> None:
        if not self.config.enable:
            return

        result_dir = f'{base_result_dir}/fpn'
        if suffix is not None:
            filename = f'fpn_{suffix}.pt'
        else:
            filename = 'fpn.pt'
        filepath = os.path.join(result_dir, filename)
        save_data = torch.load(filepath, weights_only=False, map_location=self.device)

        self.load_copy(save_data, load_optimizer_states, load_freeze_states)
