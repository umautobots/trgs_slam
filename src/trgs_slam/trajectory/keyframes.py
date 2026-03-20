from typing import Optional, Union, Literal, List, Tuple
from dataclasses import dataclass
import os
import yaml
import numpy as np
import torch

from trgs_slam.renderer.renderer import Renderer
from trgs_slam.trajectory.trajectory import TrajectoryManager, to_transformation_matrices
from trgs_slam.datasets.base_dataset import BaseDataset

@dataclass
class KeyframeConfig:
    # Alpha threshold used to determine valid depth estimates. After tracking, a depth image is rasterized at the
    # current frame and the median depth is computed across the pixels whose rasterized alpha value exceeds this
    # threshold.
    valid_depth_alpha_threshold: float = 0.95
    # Translation maximum threshold (as a fraction of the median depth). After tracking, the translation between the
    # current frame and the last keyframe is computed. The current frame is selected as a keyframe if the translation,
    # divided by the median depth observed in the current frame, is greater than this threshold (or the IoU condition is
    # met).
    translation_max: float = 0.07
    # Intersection over Union (IoU) threshold. After tracking, the IoU is computed between the set of Gaussians visible
    # to the current frame and the set of Gaussians visible to the last keyframe. The current frame will be selected as
    # a keyframe if the IoU is below this threshold and the `translation_min` threshold is met (or the `translation_max`
    # threshold is exceeded).
    iou_thresh: float = 0.90
    # Translation minimum threshold (as a fraction of the median depth). After tracking, the translation between the
    # current frame and the last keyframe is computed. The current frame will not be selected as a keyframe if the
    # translation, divided by the median depth observed in the current frame, is below this threshold.
    translation_min: float = 0.02
    # The size of the keyframe window.
    window_size: int = 6
    # The location at which to cache the keyframe images: CPU RAM, GPU RAM, or disk. Ignored for refinement.
    cache_location: Literal['cpu', 'gpu', 'disk'] = 'cpu'

@dataclass
class Keyframe:

    # The image index.
    index: int
    # The image.
    image: Optional[torch.Tensor]
    # The image timestamp.
    timestamp: torch.Tensor
    # The median rasterized depth when the keyframe was selected. Used for creating new Gaussians.
    median_depth: Optional[Union[torch.Tensor, float]] = None
    # The rasterized alpha image when the keyframe was selected. Used for creating new Gaussians.
    rasterized_alpha: Optional[torch.Tensor] = None
    # A boolean tensor indicating which Gaussians are visible to this keyframe. Used for selecting new keyframes. This
    # should be updated for the most recent keyframe after mapping.
    visible_gaussians: Optional[torch.Tensor] = None

class KeyframeManager:
    def __init__(
        self,
        config: KeyframeConfig,
        dataset: BaseDataset,
        gaussian_device: str,
        renderer: Renderer,
        trajectory_manager: TrajectoryManager,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.gaussian_device = gaussian_device
        self.renderer = renderer
        self.trajectory_manager = trajectory_manager
        self.trajectory_device = self.trajectory_manager.config.device

        self.keyframes = []
        self.window_indices = []

    def cache_kf_image(
        self,
        image: torch.Tensor
    ) -> None:
        match self.config.cache_location:
            case 'cpu':
                return image.cpu()
            case 'gpu':
                return image.to(self.gaussian_device)
            case 'disk':
                return None

    def kf_image_to_device(
        self,
        image: Union[torch.Tensor, None],
        index: int
    ) -> None:
        match self.config.cache_location:
            case 'cpu':
                return image.to(self.gaussian_device)
            case 'gpu':
                return image
            case 'disk':
                return self.dataset[index]['image'].to(self.gaussian_device)

    def add_keyframe(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.keyframes.append(Keyframe(*args, **kwargs))
        self.window_indices.append(len(self.keyframes) - 1)
        if len(self.window_indices) > self.config.window_size:
            # Remove the oldest keyframe from the window after caching its image.
            kf_exiting_window = self.keyframes[self.window_indices[0]]
            self.keyframes[self.window_indices[0]] = Keyframe(
                index=kf_exiting_window.index,
                image=self.cache_kf_image(kf_exiting_window.image),
                timestamp=kf_exiting_window.timestamp)
            del self.window_indices[0]

    @torch.no_grad()
    def keyframe_selection(
        self,
        index,
        image,
        timestamp,
    ) -> bool:
        # Render the expected depth.
        current_position_camera_in_world, current_rotation_camera_to_world = \
            self.trajectory_manager(timestamp, as_matrices=False)
        current_transformation_world_to_camera = to_transformation_matrices(
            current_position_camera_in_world,
            current_rotation_camera_to_world,
            invert=True).to(dtype=torch.float, device=self.gaussian_device)
        rasterized_depth, rasterized_alpha, info = self.renderer.rasterize(
            current_transformation_world_to_camera, render_mode='ED')

        # Compute the median depth.
        valid_depth = rasterized_alpha > self.config.valid_depth_alpha_threshold
        assert torch.any(valid_depth), 'The rasterized alpha at the new keyframe is too low to estimate the median ' \
            'depth. This either indicates that tracking has been lost or that the `valid_depth_alpha_threshold` has ' \
            'been set too high'
        median_depth = rasterized_depth[valid_depth].median()

        # Check the translation criterion.
        current_kf_position_camera_in_world, _ = \
            self.trajectory_manager(self.keyframes[-1].timestamp, as_matrices=False)
        distance = (current_position_camera_in_world[0] - current_kf_position_camera_in_world[0]).norm()
        if distance / median_depth < self.config.translation_min:
            return False
        if distance / median_depth > self.config.translation_max:
            self.add_keyframe(
                index=index,
                image=image,
                timestamp=timestamp,
                median_depth=median_depth,
                rasterized_alpha=rasterized_alpha)
            return True

        # Check the Intersection over Union (IoU) criterion.
        visible_gaussians = (info['radii'] > 0.0).all(dim=-1).squeeze()
        intersection = (visible_gaussians & self.keyframes[-1].visible_gaussians).count_nonzero()
        union = (visible_gaussians | self.keyframes[-1].visible_gaussians).count_nonzero()
        if intersection / union < self.config.iou_thresh:
            self.add_keyframe(
                index=index,
                image=image,
                timestamp=timestamp,
                median_depth=median_depth,
                rasterized_alpha=rasterized_alpha)
            return True

        return False

    def get_window_data(
        self,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        window_images = torch.cat([self.keyframes[i].image for i in self.window_indices])
        window_timestamps = torch.cat([self.keyframes[i].timestamp for i in self.window_indices])

        return self.window_indices, window_images, window_timestamps

    def get_random_data(
        self,
        num_sample: int,
        exclude_window=True,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        if exclude_window:
            if len(self.window_indices) >= len(self.keyframes):
                random_indices = []
                random_images = torch.empty((0, self.dataset.image_height, self.dataset.image_width),
                    dtype=torch.float, device=self.gaussian_device)
                random_timestamps = torch.empty(0, dtype=torch.float, device=self.trajectory_device)

                return random_indices, random_images, random_timestamps

            candidate_indices = np.arange(len(self.keyframes) - len(self.window_indices))
        else:
            candidate_indices = np.arange(len(self.keyframes))

        num_sample = min(num_sample, len(candidate_indices))
        random_indices = np.random.choice(candidate_indices, size=num_sample, replace=False).tolist()
        random_images = torch.cat([self.kf_image_to_device(self.keyframes[i].image, self.keyframes[i].index)
            for i in random_indices])
        random_timestamps = torch.cat([self.keyframes[i].timestamp for i in random_indices])

        return random_indices, random_images, random_timestamps

    def update_visible_gaussians(
        self,
    ) -> None:
        current_kf_timestamp = self.keyframes[-1].timestamp
        current_kf_transformation_world_to_camera = self.trajectory_manager(current_kf_timestamp, invert=True).to(
            dtype=torch.float, device=self.gaussian_device)
        _, _, info = self.renderer.rasterize(current_kf_transformation_world_to_camera)
        self.keyframes[-1].visible_gaussians = (info['radii'] > 0.0).all(dim=-1).squeeze()

    def save(
        self,
        base_result_dir: str,
        suffix: Optional[str] = None,
    ) -> None:
        result_dir = f'{base_result_dir}/keyframes'
        if suffix is not None:
            filename = f'keyframes_{suffix}.yaml'
        else:
            filename = 'keyframes.yaml'
        os.makedirs(result_dir, exist_ok=True)
        filepath = os.path.join(result_dir, filename)

        keyframes_dict = {
            'indices': [kf.index for kf in self.keyframes],
            'timestamps': torch.cat([kf.timestamp for kf in self.keyframes]).tolist()}
        with open(filepath, 'w', encoding='utf8') as file:
            yaml.dump(keyframes_dict, file)

    def load(
        self,
        base_result_dir: str,
        suffix: Optional[str] = None,
    ) -> None:
        result_dir = f'{base_result_dir}/keyframes'
        if suffix is not None:
            filename = f'keyframes_{suffix}.yaml'
        else:
            filename = 'keyframes.yaml'
        filepath = os.path.join(result_dir, filename)

        with open(filepath, 'r') as file:
            keyframes_dict = yaml.safe_load(file)

        for index, timestamp in zip(keyframes_dict['indices'], keyframes_dict['timestamps']):
            if self.config.cache_location == 'disk':
                image = None
            else:
                image = self.cache_kf_image(self.dataset[index]['image'].unsqueeze(0))
            timestamp = torch.tensor([timestamp], dtype=torch.double, device=self.trajectory_device)
            self.add_keyframe(index=index, image=image, timestamp=timestamp)

        self.window_indices = np.arange(len(self.keyframes) - self.config.window_size, len(self.keyframes)).tolist()
        for i in self.window_indices:
            self.keyframes[i].image = self.kf_image_to_device(self.keyframes[i].image, self.keyframes[i].index)

        self.update_visible_gaussians
