from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import os
import copy

from sklearn.neighbors import NearestNeighbors
import torch

from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat.optimizers import SelectiveAdam

from trgs_slam.gaussians.strategy import SLAMStrategy

@dataclass
class GaussianConfig:
    # Device on which to store the Gaussian parameters and perform image based computations (e.g., computing
    # the radiometric loss).
    device: str = 'cuda:0'
    # The initial opacity given to newly created Gaussians.
    initial_opacity: float = 0.1
    # Learning rate for Gaussian means (3D positions).
    means_lr: float = 1.6e-3
    # Learning rate for Gaussian scale factors.
    scales_lr: float = 5e-3
    # Learning rate for Gaussian opacities.
    opacities_lr: float = 5e-2
    # Learning rate for Gaussian orientations (quaternions).
    quats_lr: float = 1e-3
    # Learning rate for Gaussian intensities.
    intensities_lr: float = 2.5e-3
    # Whether to use the visible/sparse Adam method from "Taming 3DGS".
    visible_adam: bool = False
    # Densification options.
    strategy: SLAMStrategy = field(default_factory=SLAMStrategy)

class GaussianManager:
    def __init__(
        self,
        config: GaussianConfig
    ) -> None:
        self.config = config
        self.params = None
        self.optimizers = None

        # Initialize the densification strategy.
        self.strategy_state = self.config.strategy.initialize_state()

    def initialize_optimizers(
        self,
    ) -> None:
        lr_dict = {
            'means': self.config.means_lr,
            'scales': self.config.scales_lr,
            'quats': self.config.quats_lr,
            'opacities': self.config.opacities_lr,
            'intensities': self.config.intensities_lr}

        if self.config.visible_adam:
            self.optimizers = {
                name: SelectiveAdam(
                    [{'params': self.params[name], 'lr': lr, 'name': name}],
                    eps=1e-15, betas=(0.9, 0.999))
                for name, lr in lr_dict.items()}
        else:
            self.optimizers = {
                name: torch.optim.Adam([{'params': self.params[name], 'lr': lr, 'name': name}])
                for name, lr in lr_dict.items()}

    @torch.no_grad()
    def add_gaussians(
        self,
        gaussian_dict: Dict[str, torch.Tensor],
    ) -> None:
        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(torch.cat([p, gaussian_dict[name]]), requires_grad=p.requires_grad)

        num_gaussians = len(gaussian_dict['opacities'])
        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.cat([v, torch.zeros((num_gaussians, *v.shape[1:]), device=v.device, dtype=v.dtype)])

        # Update the parameters and the state in the optimizers.
        _update_param_with_optimizer(param_fn, optimizer_fn, self.params, self.optimizers)

        # Update the strategy state.
        for k, v in self.strategy_state.items():
            if isinstance(v, torch.Tensor):
                self.strategy_state[k] = \
                    torch.cat((v, torch.zeros((num_gaussians, *v.shape[1:]), device=v.device, dtype=v.dtype)))

    @torch.no_grad()
    def create_gaussians(
        self,
        image: torch.Tensor,
        rasterized_alpha: torch.Tensor,
        median_depth: torch.Tensor,
        new_camera_matrix: torch.Tensor,
        transformation_camera_to_world: torch.Tensor,
        max_num_gaussians: int,
        alpha_threshold: float = 1.0,
        skip_threshold: float = 0.0,
    ) -> None:
        # Determine candidate pixels for back projection (parts of the FOV where the map is poorly reconstructed).
        y_coords, x_coords = torch.nonzero(rasterized_alpha < alpha_threshold, as_tuple=True)
        candidate_fraction = len(x_coords) / (image.shape[0] * image.shape[1])
        if candidate_fraction < skip_threshold:
            return

        # Randomly select pixels to back project.
        if max_num_gaussians > image.shape[0] * image.shape[1]:
            max_num_gaussians = image.shape[0] * image.shape[1]
        num_gaussians = int(max_num_gaussians * candidate_fraction)
        indices = torch.randperm(len(x_coords), device=self.config.device)[:num_gaussians]

        # Back project the selected pixels with the median depth to initialize the Gaussian means and transform them
        # into the world frame.
        homogeneous_pixel_coords = torch.stack([x_coords[indices], y_coords[indices], torch.ones_like(indices)])
        means_camera = torch.linalg.inv(new_camera_matrix) @ homogeneous_pixel_coords.float() * median_depth
        means_world = (transformation_camera_to_world[:3, :3] @ means_camera +
            transformation_camera_to_world[:3, 3].unsqueeze(-1)).T

        # Initialize the scale of each Gaussian to be the average distance of its 3 nearest neighbors.
        means_world_np = means_world.cpu().numpy()
        model = NearestNeighbors(n_neighbors=4, metric='euclidean').fit(means_world_np)
        distances, _ = model.kneighbors(means_world_np)
        distances = torch.from_numpy(distances).to(means_world)
        average_distance  = torch.sqrt((distances[:, 1:] ** 2).mean(dim=-1))
        scales = torch.log(average_distance).unsqueeze(-1).repeat(1, 3)

        # Randomly initialize the Gaussian orientations.
        quats = torch.rand((num_gaussians, 4), device=self.config.device)

        # Initialize the opacities to the given value.
        opacities = torch.logit(torch.full((num_gaussians,), self.config.initial_opacity, \
            dtype=torch.float, device=self.config.device))

        # Initialize the Gaussian intensities with the corresponding pixel values in the image.
        intensities = torch.logit(image[y_coords[indices], x_coords[indices]].unsqueeze(-1))
        gaussian_dict = {
            'means': means_world.contiguous(),
            'scales': scales.contiguous(),
            'quats': quats.contiguous(),
            'opacities': opacities.contiguous(),
            'intensities': intensities.contiguous()}

        # Initialize or update the Gaussian parameters and optimizers.
        if self.params is None:
            self.params = torch.nn.ParameterDict(gaussian_dict)
            self.initialize_optimizers()
        else:
            self.add_gaussians(gaussian_dict)

    def strategy_step_pre_backward(
        self,
        step: int,
        info: Dict[str, Any],
    ) -> None:
        self.config.strategy.step_pre_backward(self.params, self.optimizers, self.strategy_state, step, info)

    def strategy_step_post_backward(
        self,
        step: int,
        info: Dict[str, Any],
        vdb_resetting: bool = False,
        densifying: bool = True,
    ) -> bool:
        return self.config.strategy.step_post_backward(self.params, self.optimizers, self.strategy_state, step,
            info, vdb_resetting, densifying)

    def zero_grad(
        self
    ) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    def step(
        self,
        info: Optional[Dict] = None,
    ) -> None:
        if self.config.visible_adam:
            assert info is not None, 'The visible Adam optimizer requires the rasterization metadata to compute a ' \
                'visibility mask.'
            visibility_mask = (info['radii'] > 0).all(-1).any(0)
            for optimizer in self.optimizers.values():
                optimizer.step(visibility_mask)
        else:
            for optimizer in self.optimizers.values():
                optimizer.step()

    def freeze(
        self,
    ) -> None:
        for parameter in self.params.values():
            parameter.requires_grad = False

    def unfreeze(
        self,
    ) -> None:
        for parameter in self.params.values():
            parameter.requires_grad = True

    def get_copy(
        self,
        save_strategy_state: bool = True,
        save_optimizer_states: bool = True,
        save_freeze_states: bool = True,
    ) -> Dict[str, Any]:
        save_data = {'parameter_states': copy.deepcopy(self.params.state_dict())}

        if save_strategy_state:
            save_data['strategy_state'] = copy.deepcopy(self.strategy_state)

        if save_optimizer_states:
            save_data['optimizer_states'] = {}
            for name, optimizer in self.optimizers.items():
                save_data['optimizer_states'][name] = copy.deepcopy(optimizer.state_dict())

        if save_freeze_states:
            save_data['freeze_states'] = {}
            for name, parameter in self.params.items():
                save_data['freeze_states'][name] = parameter.requires_grad

        return save_data

    def save(
        self,
        base_result_dir: str,
        suffix: Optional[str] = None,
        save_strategy_state: bool = True,
        save_optimizer_states: bool = True,
        save_freeze_states: bool = True,
    ) -> None:
        result_dir = f'{base_result_dir}/gaussians'
        if suffix is not None:
            filename = f'gaussians_{suffix}.pt'
        else:
            filename = 'gaussians.pt'
        os.makedirs(result_dir, exist_ok=True)
        filepath = os.path.join(result_dir, filename)

        torch.save(self.get_copy(save_strategy_state, save_optimizer_states, save_freeze_states), filepath)

    def load_copy(
        self,
        save_data: Dict[str, Any],
        load_strategy_state: bool = True,
        load_optimizer_states: bool = True,
        load_freeze_states: bool = True,
    ) -> None:
        self.params = torch.nn.ParameterDict(save_data['parameter_states'])
        self.initialize_optimizers()

        if load_strategy_state:
            if 'strategy_state' in save_data:
                self.strategy_state = save_data['strategy_state']
            else:
                print('Warning: the strategy state was not saved. The strategy state has been reinitialized.')
                self.strategy_state = self.config.strategy.initialize_state()
        else:
            self.strategy_state = self.config.strategy.initialize_state()

        if load_optimizer_states:
            if 'optimizer_states' in save_data:
                for name, optimizer in self.optimizers.items():
                    optimizer.load_state_dict(save_data['optimizer_states'][name])
            else:
                print('Warning: no Gaussian optimizer states were saved. The Gaussian optimizers have been initialized '
                    'with default states')

        if load_freeze_states:
            if 'freeze_states' in save_data:
                for name, parameter in self.params.items():
                    parameter.requires_grad = save_data['freeze_states'][name]
            else:
                print('Warning: no freeze states were saved. The parameters have been initialized with '
                    'requires_grad == True')

    def load(
        self,
        base_result_dir: str,
        suffix: Optional[str] = None,
        load_strategy_state: bool = True,
        load_optimizer_states: bool = True,
        load_freeze_states: bool = True,
    ) -> None:
        result_dir = f'{base_result_dir}/gaussians'
        if suffix is not None:
            filename = f'gaussians_{suffix}.pt'
        else:
            filename = 'gaussians.pt'
        filepath = os.path.join(result_dir, filename)
        save_data = torch.load(filepath, weights_only=False, map_location=self.config.device)

        self.load_copy(save_data, load_strategy_state, load_optimizer_states, load_freeze_states)
