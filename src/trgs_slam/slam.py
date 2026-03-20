from typing import Literal, Optional
from dataclasses import dataclass, field, asdict
import os

import yaml
import torch

from fused_ssim import fused_ssim

from trgs_slam.gaussians.gaussians import GaussianConfig, GaussianManager
from trgs_slam.renderer.renderer import RendererConfig, Renderer
from trgs_slam.trajectory.trajectory import TrajectoryConfig, TrajectoryManager
from trgs_slam.trajectory.keyframes import KeyframeConfig, KeyframeManager
from trgs_slam.trajectory.imu_utils import analytical_imu_initialization
from trgs_slam.tools.viewer import SLAMViewerConfig, SLAMViewer
from trgs_slam.tools.nested_timer import NestedTimer
from trgs_slam.datasets.dataset_configs import DatasetsUnion
from trgs_slam.datasets.trnerf_dataset import TRNeRFDatasetConfig
from trgs_slam.tools.metrics import ATECalculator

@dataclass
class TrackingConfig:
    # Maximum number of optimization steps for tracking.
    tracking_max_num_iter: int = 100
    # Translation convergence threshold (unitless). The current frame's unscaled position estimate will be considered
    # converged if it changed by less than this threshold in the last optimization step. Tracking stops if the current
    # frame's position and rotation estimates have converged.
    convergence_position_thresh: float = 1e-4
    # Rotation convergence threshold (in radians). The current frame's rotation estimate will be considered converged
    # if it changed by less than this threshold in the last optimization step. Tracking stops if the current frame's
    # position and rotation estimates have converged.
    convergence_rotation_thresh: float = 1e-4

@dataclass
class MappingConfig:
    # Number of Gaussians created for map initialization.
    map_init_num_gaussians: int = 10_000
    # Number of optimization steps for map initialization.
    map_init_num_iter: int = 1_000
    # Alpha threshold used for back projection. When a new keyframe is added, new Gaussians are created by back
    # projecting pixels that are randomly selected from the set of pixels whose rendered alpha value is below this
    # threshold.
    back_project_alpha_threshold: float = 0.95
    # Maximum number of Gaussians created when a new keyframe is added. The actual number of Gaussians created will be
    # this number multiplied by the fraction of pixels whose rendered alpha value is below
    # `back_project_alpha_threshold`.
    new_kf_max_num_gaussians: int = 5_000
    # Skip new Gaussians threshold. If the fraction of pixels whose rendered alpha value is below
    # `back_project_alpha_threshold` is below this threshold, no new Gaussians will be created for the new keyframe.
    skip_new_gaussians_threshold: float = 0.02
    # Number of optimization steps for mapping when a new keyframe is added.
    new_kf_num_iter: int = 80
    # How many past keyframes to randomly select and include in mapping when a new keyframe is added.
    num_random_kfs: int = 4
    # Relative weight to apply to randomly selected keyframes in mapping when a new keyframe is added.
    random_kf_relative_weight: float = 6.0
    # Additional factor to apply to position and rotation control point gradients corresponding to the keyframe window
    # in mapping when a new keyframe is added.
    window_gradient_factor: float = 10.0
    # Relocalization SSIM threshold. Relocalization is performed if the SSIM is below this threshold after mapping
    # with a new keyframe.
    reloc_ssim_thresh: float = 0.92

@dataclass
class IMUInitConfig:
    # When to initialize the IMU (in seconds after the first frame).
    imu_init_time: float = 10.0
    # Number of optimization steps for mapping in the second stage of IMU initialization.
    imu_init_second_stage_num_iter: int = 400
    # Period between IMU updates after initialization (in seconds).
    imu_update_period: float = 10.0
    # Rate at which poses are sampled for IMU initialization/updates (Hz).
    imu_update_pose_rate: float = 2.0

@dataclass
class WarmupConfig:
    # How many keyframes to process before the position and rotation splines are fit to predicted poses when they are
    # extended (prior to this, the spline is extended simply by duplicating the most recent control point).
    predict_poses_warmup: int = 2
    # How many keyframes to process before motion artifacts are modeled during rendering.
    model_motion_artifacts_warmup: int = 20
    # How many keyframes to process before applying view-diversity-based opacity resetting during densification.
    vdb_reset_warmup: int = 40
    # How many keyframes to process before relocalization is enabled.
    relocalize_warmup: int = 150

@dataclass
class TimerConfig:
    # Whether to enable the timer.
    enable: bool = False
    # Whether to force synchronization (with torch.cuda.synchronize()) after every timed subtask. This is necessary to
    # get accurate timing results, but comes at a slight cost to performance. This forced synchronization is not
    # performed if the timer is disabled.
    force_cuda_sync: bool = True
    # Whether to limit printed subtask timing statistics to just the mean or to additionally print the median, min, and
    # max.
    print_mean_only: bool = True
    # Print timing summary after map initialization.
    print_map_init_summary: bool = False
    # Print timing summary after each frame tracked.
    print_tracking_summary: bool = False
    # Printing timing summary after each keyframe mapped.
    print_mapping_summary: bool = False
    # Pring timing summary after IMU initialization and each IMU update.
    print_imu_init_summary: bool = False

@dataclass
class SLAMConfig:
    # The directory to save results in.
    result_dir: Optional[str] = None
    # How frequently to save results (in units of keyframes). The final result will also be saved.
    save_interval: int = 50
    # The directory to load results from. If provided, `load_suffix` must also be provided.
    load_dir: Optional[str] = None
    # The suffix to load (typically, this is an image index).
    load_suffix: Optional[str] = None
    # Whether to load the densification strategy state. Necessary for resuming SLAM, but not for refinement.
    load_strategy_state: bool = True
    # Whether to load the optimizer states. Necessary for resuming SLAM, but not for refinement.
    load_optimizer_states: bool = True
    # Whether to load the freeze states. Necessary for resuming SLAM, but not for refinement.
    load_freeze_states: bool = True
    # Tracking options.
    tracking_config: TrackingConfig = field(default_factory=TrackingConfig)
    # Mapping options.
    mapping_config: MappingConfig = field(default_factory=MappingConfig)
    # Warmup options.
    warmup_config: WarmupConfig = field(default_factory=WarmupConfig)
    # IMU initialization options.
    imu_init_config: IMUInitConfig = field(default_factory=IMUInitConfig)
    # Gaussian options.
    gaussian_config: GaussianConfig = field(default_factory=GaussianConfig)
    # Renderer options.
    renderer_config: RendererConfig = field(default_factory=RendererConfig)
    # Trajectory options.
    trajectory_config: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    # Keyframe options.
    keyframe_config: KeyframeConfig = field(default_factory=KeyframeConfig)
    # Viewer options.
    viewer_config: SLAMViewerConfig = field(default_factory=SLAMViewerConfig)
    # Timer options.
    timer_config: TimerConfig = field(default_factory=TimerConfig)
    # Dataset options.
    dataset_config: DatasetsUnion = field(default_factory=TRNeRFDatasetConfig)

@dataclass
class SLAMState:
    # Whether the position and rotation splines are being fit to predicted poses when they are extended.
    predicting_poses: bool = False
    # Whether motion artifacts are being modeled during rendering.
    modeling_motion_artifacts: bool = False
    # Whether view-diversity-based opacity resetting is being applied during densification.
    vdb_resetting: bool = False
    # Whether Gaussian densification is being performed periodically during mapping.
    densifying: bool = False
    # Whether relocalization is being performed.
    relocalizing: bool = False
    # The last time the IMU parameters were updated.
    last_update_time: Optional[float] = None
    # Global mapping step count.
    global_mapping_step: int = 0
    # Current image index.
    current_index: int = 0

class SLAM:
    def __init__(self, config: SLAMConfig) -> None:
        self.config = config
        self.gaussian_device = self.config.gaussian_config.device
        self.trajectory_device = self.config.trajectory_config.device

        # Create the output directory and write the config to it.
        if self.config.result_dir is not None:
            os.makedirs(self.config.result_dir, exist_ok=True)
            with open(f'{self.config.result_dir}/config.yaml', 'w', encoding='utf8') as file:
                yaml.dump(self.config, file)

        # Initialize the dataset and data loader iterator.
        self.dataset = self.config.dataset_config._dataset_class(self.config.dataset_config)
        self.new_camera_matrix = torch.from_numpy(self.dataset.new_camera_matrix).to(
            dtype=torch.float, device=self.config.gaussian_config.device)
        if self.dataset.imu_available:
            self.transformation_imu_to_camera = torch.from_numpy(self.dataset.transformation_imu_to_camera).to(
                dtype=torch.double, device=self.trajectory_device)
        data_loader = torch.utils.data.DataLoader(self.dataset)
        self.data_loader_iterator = iter(data_loader)

        # Initialize the SLAM state.
        self.slam_state = SLAMState()

        # Read in the first image.
        if self.dataset.imu_available:
            self.linear_accels_meas_imu = \
                torch.empty((0, 3), dtype=torch.double, device=self.trajectory_device)
            self.angular_vels_meas_imu = \
                torch.empty((0, 3), dtype=torch.double, device=self.trajectory_device)
            self.timestamps_imu = torch.empty(0, dtype=torch.double, device=self.trajectory_device)
        self.read_next_image()

        # Initialize the Gaussian manager.
        self.gaussian_manager = GaussianManager(self.config.gaussian_config)

        # Initialize the renderer.
        self.renderer = Renderer(
            self.config.renderer_config,
            self.dataset,
            self.gaussian_manager,
            self.trajectory_device,
            self.current_timestamp.item())

        # Initialize the trajectory manager.
        if self.config.renderer_config.num_rasters > 1:
            time_start = (self.current_timestamp + self.renderer.time_offsets.min()).item()
            self.trajectory_manager = TrajectoryManager(
                self.config.trajectory_config,
                self.dataset,
                time_start)
            next_latest_time_eval = (self.current_timestamp + self.renderer.time_offsets.max()).item()
            self.trajectory_manager.extend_splines(next_latest_time_eval, predict_poses=False, enable_timing=False)
        else:
            self.trajectory_manager = TrajectoryManager(
                self.config.trajectory_config,
                self.dataset,
                self.current_timestamp.item())

        # Initialize the keyframe manager.
        self.keyframe_manager = KeyframeManager(
            self.config.keyframe_config,
            self.dataset,
            self.gaussian_device,
            self.renderer,
            self.trajectory_manager)

        # Initialize the viewer.
        self.viewer = SLAMViewer(
            self.config.viewer_config,
            self.renderer,
            self.gaussian_manager,
            self.trajectory_manager,
            self.keyframe_manager,
            self.dataset)

        # Initialize the map or load in a result.
        if self.config.load_dir is None:
            self.initialize_map(print_timing_summary=self.config.timer_config.print_map_init_summary)
        else:
            self.load()

    def run(self) -> None:
        # Process all remaining data.
        while self.read_next_image():
            # Update warmup states.
            num_kfs = len(self.keyframe_manager.keyframes)
            self.slam_state.predicting_poses = num_kfs >= self.config.warmup_config.predict_poses_warmup
            self.slam_state.modeling_motion_artifacts = \
                (self.config.renderer_config.num_rasters > 1) and \
                (num_kfs >= self.config.warmup_config.model_motion_artifacts_warmup)
            self.slam_state.vdb_resetting = num_kfs >= self.config.warmup_config.vdb_reset_warmup

            # Track the current frame.
            selected_as_keyframe = self.track(print_timing_summary=self.config.timer_config.print_tracking_summary)

            if selected_as_keyframe:
                # Run mapping for new keyframes.
                self.map_new_keyframe(print_timing_summary=self.config.timer_config.print_mapping_summary)

                # Update the IMU parameters periodically.
                do_imu_update = self.dataset.imu_available and (not self.slam_state.relocalizing)
                if not self.trajectory_manager.imu_frame:
                    time_since_update = self.keyframe_manager.keyframes[-1].timestamp - \
                        self.keyframe_manager.keyframes[0].timestamp
                    do_imu_update = do_imu_update and (time_since_update >= self.config.imu_init_config.imu_init_time)
                else:
                    time_since_update = self.keyframe_manager.keyframes[-1].timestamp - self.slam_state.last_update_time
                    do_imu_update = do_imu_update and \
                        (time_since_update >= self.config.imu_init_config.imu_update_period)
                if do_imu_update:
                    self.imu_initialization(print_timing_summary=self.config.timer_config.print_imu_init_summary)
                    self.slam_state.last_update_time = self.keyframe_manager.keyframes[-1].timestamp.item()

                # Save checkpoints periodically.
                if len(self.keyframe_manager.keyframes) % self.config.save_interval == 0:
                    self.save()

    def read_next_image(
        self
    ) -> None:
        try:
            data = next(self.data_loader_iterator)
        except StopIteration:
            return False

        self.slam_state.current_index = data['index'].item()
        if self.slam_state.current_index > 0:
            self.previous_timestamp = self.current_timestamp
        self.current_image = data['image'].to(self.gaussian_device)
        self.current_timestamp = data['image_timestamp'].to(self.trajectory_device)
        if self.dataset.imu_available:
            self.linear_accels_meas_imu = torch.cat([self.linear_accels_meas_imu, data['linear_accels'].to(
                dtype=torch.double, device=self.trajectory_device).squeeze(0)])
            self.angular_vels_meas_imu = torch.cat([self.angular_vels_meas_imu, data['angular_vels'].to(
                dtype=torch.double, device=self.trajectory_device).squeeze(0)])
            self.timestamps_imu = torch.cat([self.timestamps_imu, data['imu_timestamps'].to(
                self.trajectory_device).squeeze(0)])

        return True

    def map(
        self,
        mode: Literal['map_init', 'new_keyframe', 'imu_init'],
        num_iter: int,
        max_num_gaussians: int = 5_000,
    ) -> None:
        if mode == 'map_init' or mode == 'new_keyframe':
            # Create new Gaussians for the current keyframe.
            with NestedTimer('Gaussian Creation'):
                with torch.no_grad():
                    current_transformation_camera_to_world = \
                        self.trajectory_manager(self.keyframe_manager.keyframes[-1].timestamp).to(
                        dtype=torch.float, device=self.gaussian_device)
                self.gaussian_manager.create_gaussians(
                    self.keyframe_manager.keyframes[-1].image.squeeze(),
                    self.keyframe_manager.keyframes[-1].rasterized_alpha.squeeze(),
                    self.keyframe_manager.keyframes[-1].median_depth,
                    self.new_camera_matrix,
                    current_transformation_camera_to_world.squeeze(),
                    max_num_gaussians,
                    self.config.mapping_config.back_project_alpha_threshold,
                    self.config.mapping_config.skip_new_gaussians_threshold)

        # Compile keyframe window data.
        if mode == 'map_init' or mode == 'new_keyframe':
            kf_window_indices, kf_window_images, kf_window_timestamps = self.keyframe_manager.get_window_data()

        step = 0
        while step < num_iter:
            with NestedTimer('Mapping Iteration'):
                self.viewer.pre_loop()

                # Zero the gradients.
                with NestedTimer('Zero Gradients'):
                    self.gaussian_manager.zero_grad()
                    self.renderer.zero_grad()
                    self.trajectory_manager.zero_grad()

                with NestedTimer('Select Data'):
                    num_weighted_kfs = 0
                    if mode == 'map_init':
                        # Use only the data from the first frame / keyframe.
                        kf_indices, kf_images, kf_timestamps = kf_window_indices, kf_window_images, kf_window_timestamps
                    elif mode == 'new_keyframe':
                        # Compile randomly selected keyframe data and combine it the keyframe window data.
                        random_kf_indices, random_kf_images, random_kf_timestamps = \
                            self.keyframe_manager.get_random_data(self.config.mapping_config.num_random_kfs)
                        kf_indices = random_kf_indices + kf_window_indices
                        kf_images = torch.cat([random_kf_images, kf_window_images])
                        kf_timestamps = torch.cat([random_kf_timestamps, kf_window_timestamps])

                        num_weighted_kfs = len(random_kf_indices)
                    elif mode == 'imu_init':
                        # Compile randomly selected keyframe data.
                        kf_num_sample = \
                            self.keyframe_manager.config.window_size + self.config.mapping_config.num_random_kfs
                        kf_indices, kf_images, kf_timestamps = \
                            self.keyframe_manager.get_random_data(kf_num_sample, exclude_window=False)

                    if self.dataset.imu_available:
                        if mode == 'new_keyframe':
                            # Select IMU data for each keyframe.
                            prev_kf_timestamps = \
                                torch.cat([self.keyframe_manager.keyframes[max(0, i-1)].timestamp for i in kf_indices])
                            if self.slam_state.modeling_motion_artifacts:
                                imu_start_times = \
                                    torch.minimum(prev_kf_timestamps, kf_timestamps + self.renderer.time_offsets.min())
                                imu_end_times = kf_timestamps + self.renderer.time_offsets.max()
                            else:
                                imu_start_times = prev_kf_timestamps
                                imu_end_times = kf_timestamps
                            imu_selection_mask = \
                                ((self.timestamps_imu >= imu_start_times.unsqueeze(-1)) &
                                (self.timestamps_imu <= imu_end_times.unsqueeze(-1))).any(dim=0)
                            selected_linear_accels = self.linear_accels_meas_imu[imu_selection_mask]
                            selected_angular_vels = self.angular_vels_meas_imu[imu_selection_mask]
                            selected_imu_timestamps = self.timestamps_imu[imu_selection_mask]
                        elif mode == 'imu_init':
                            # Use all IMU data (up to the latest evaluated timestamp).
                            imu_selection_mask = self.timestamps_imu <= self.trajectory_manager.latest_time_eval
                            selected_linear_accels = self.linear_accels_meas_imu[imu_selection_mask]
                            selected_angular_vels = self.angular_vels_meas_imu[imu_selection_mask]
                            selected_imu_timestamps = self.timestamps_imu[imu_selection_mask]

                # Render the images.
                output_images, info, _, _ = self.renderer.render(
                    kf_timestamps,
                    self.trajectory_manager,
                    self.slam_state.modeling_motion_artifacts)

                # Retain necessary gradients for the densification strategy.
                with NestedTimer('Strat Pre Backward'):
                    self.gaussian_manager.strategy_step_pre_backward(self.slam_state.global_mapping_step, info)

                with NestedTimer('Loss Computation'):
                    # Compute the radiometric loss.
                    with NestedTimer('Radiometric Loss'):
                        if num_weighted_kfs > 0:
                            weights = torch.ones(len(kf_indices), dtype=torch.float, device=self.gaussian_device)
                            weights[:num_weighted_kfs] *= self.config.mapping_config.random_kf_relative_weight
                            weights = weights / weights.mean()
                            loss = torch.abs(
                                (output_images['rendered_images'] - kf_images) * weights.view(-1, 1, 1)).mean()

                            if self.slam_state.modeling_motion_artifacts:
                                latest_random_timestamp = random_kf_timestamps.max() + self.renderer.time_offsets.max()
                            else:
                                latest_random_timestamp = random_kf_timestamps.max()
                            weight_random = weights[0].to(self.trajectory_device)
                            weight_window = weights[-1].to(self.trajectory_device)
                        else:
                            loss = torch.nn.functional.l1_loss(output_images['rendered_images'], kf_images)

                    # Compute the FPN pixelwise offsets loss.
                    with NestedTimer('FPN Loss'):
                        loss += self.renderer.get_loss(kf_timestamps)

                    # Compute the IMU loss with the selected IMU measurements.
                    with NestedTimer('IMU Loss'):
                        if self.dataset.imu_available and (mode == 'new_keyframe' or mode == 'imu_init'):
                            if num_weighted_kfs > 0:
                                loss += self.trajectory_manager.get_loss(
                                    selected_linear_accels,
                                    selected_angular_vels,
                                    selected_imu_timestamps,
                                    latest_random_timestamp,
                                    weight_random,
                                    weight_window,
                                    self.slam_state.relocalizing)
                            else:
                                loss += self.trajectory_manager.get_loss(
                                    selected_linear_accels,
                                    selected_angular_vels,
                                    selected_imu_timestamps,
                                    use_reloc_weights=self.slam_state.relocalizing)

                with NestedTimer('Backward Pass'):
                    # Compute the gradients.
                    loss.backward()

                # Weight the control point and Gaussian gradients.
                with NestedTimer('Manual Gradient Manipulation'):
                    if num_weighted_kfs > 0:
                        first_window_index = \
                            torch.floor((latest_random_timestamp - self.trajectory_manager.time_start) /
                                self.trajectory_manager.config.pose_knot_interval).int() + \
                            self.trajectory_manager.config.pose_spline_order

                        pos_cp_grad = self.trajectory_manager.trajectory.position_spline.control_points.grad
                        random_mask = pos_cp_grad._indices()[0] < first_window_index
                        pos_cp_grad._values()[random_mask] /= weight_random
                        pos_cp_grad._values()[~random_mask] /= \
                            (weight_window / self.config.mapping_config.window_gradient_factor)

                        rot_cp_grad = self.trajectory_manager.trajectory.rotation_spline.control_points.grad
                        random_mask = rot_cp_grad._indices()[0] < first_window_index
                        rot_cp_grad._values()[random_mask] /= weight_random
                        rot_cp_grad._values()[~random_mask] /= \
                            (weight_window / self.config.mapping_config.window_gradient_factor)

                        if self.slam_state.modeling_motion_artifacts:
                            num_weighted_rasters = num_weighted_kfs * self.renderer.config.num_rasters
                        else:
                            num_weighted_rasters = num_weighted_kfs
                        count_random = (info['radii'] > 0.0).all(dim=-1)[:num_weighted_rasters].sum(dim=0)
                        count_window = (info['radii'] > 0.0).all(dim=-1)[num_weighted_rasters:].sum(dim=0)
                        visible_mask = (count_random > 0) | (count_window > 0)
                        fraction_random = \
                            count_random[visible_mask] / (count_random[visible_mask] + count_window[visible_mask])
                        for param in self.gaussian_manager.params:
                            self.gaussian_manager.params[param].grad[visible_mask] /= \
                                (weight_random * fraction_random + weight_window * (1 - fraction_random)).view(
                                    -1, *([1] * (self.gaussian_manager.params[param].grad[visible_mask].ndim - 1)))

                with NestedTimer('Optimizer Updates'):
                    # Update the parameters.
                    with NestedTimer('Gaussian Update'):
                        self.gaussian_manager.step(info)
                    with NestedTimer('FPN Update'):
                        self.renderer.step()
                    with NestedTimer('Trajectory Update'):
                        self.trajectory_manager.step()

                # Accumulate information or perform densification and pruning.
                densified = self.gaussian_manager.strategy_step_post_backward(
                    self.slam_state.global_mapping_step,
                    info,
                    self.slam_state.vdb_resetting,
                    self.slam_state.densifying)

                # Update the viewer.
                if mode == 'map_init':
                    self.viewer.post_loop(
                        self.slam_state.global_mapping_step,
                        render=output_images['rendered_images'][-1])
                elif mode == 'new_keyframe':
                    self.viewer.post_loop(
                        self.slam_state.global_mapping_step,
                        render=output_images['rendered_images'][-1],
                        fpn_time=self.keyframe_manager.keyframes[-1].timestamp)
                elif mode == 'imu_init':
                    self.viewer.post_loop(self.slam_state.global_mapping_step)

                # Update the step counts.
                self.slam_state.global_mapping_step += 1
                step += 1
                if densified:
                    # Reset the step counter after densification
                    step = 0

        # Update which Gaussians are visible to the current keyframe.
        self.keyframe_manager.update_visible_gaussians()

    @NestedTimer.timed_function('Map Initialization')
    def initialize_map(
        self,
    ) -> None:
            # Add the initial keyframe, setting the median depth and rasterized alpha such that Gaussians are
            # initialized in a plane 1 unit (unscaled) in front of the camera.
            self.keyframe_manager.add_keyframe(
                index=self.slam_state.current_index,
                image=self.current_image,
                timestamp=self.current_timestamp,
                median_depth=torch.tensor([1.0], dtype=torch.float, device=self.gaussian_device),
                rasterized_alpha=torch.zeros_like(self.current_image))

            # Update the viewer with the current image and the initial pose.
            self.viewer.update_frame(self.current_image, enable_timing=False)
            self.viewer.update_viewer_kfs(mode='mapping', enable_timing=False)

            # Run mapping with the trajectory and FPN offsets frozen.
            print('Running map initialization...')
            self.trajectory_manager.freeze(poses=True)
            self.renderer.freeze()
            self.map(
                'map_init',
                self.config.mapping_config.map_init_num_iter,
                self.config.mapping_config.map_init_num_gaussians)
            self.trajectory_manager.unfreeze(poses=True)
            self.renderer.unfreeze()
            print('Map initialization complete.')

            # Enable densification for subsequent mapping stages.
            self.slam_state.densifying = True

    @NestedTimer.timed_function('Mapping New Keyframe')
    def map_new_keyframe(
        self,
    ) -> None:
            with NestedTimer('Data Copying'):
                # Make a complete copy.
                gaussians_copy = self.gaussian_manager.get_copy()
                trajectory_copy = self.trajectory_manager.get_copy()
                renderer_copy = self.renderer.get_copy()
                global_mapping_step_copy = self.slam_state.global_mapping_step

            # Run mapping.
            self.map(
                'new_keyframe',
                self.config.mapping_config.new_kf_num_iter,
                self.config.mapping_config.new_kf_max_num_gaussians)

            # Compute SSIM.
            with torch.no_grad():
                current_kf_image_clean = self.renderer.fpn_manager(
                    self.keyframe_manager.keyframes[-1].image,
                    self.keyframe_manager.keyframes[-1].timestamp,
                    add=False)
                output_images, _, _, _ = self.renderer.render(
                    self.keyframe_manager.keyframes[-1].timestamp,
                    self.trajectory_manager,
                    self.slam_state.modeling_motion_artifacts,
                    enable_timing=False)
                current_ssim = fused_ssim(
                    current_kf_image_clean.unsqueeze(-1).permute(0, 3, 1, 2),
                    output_images['rendered_images_clean'].unsqueeze(-1).permute(0, 3, 1, 2),
                    padding='valid')
            print(f'Mapping complete: '
                f'image index {self.slam_state.current_index}, '
                f'KF index {len(self.keyframe_manager.keyframes) - 1}, '
                f'SSIM {current_ssim}')

            if current_ssim < self.config.mapping_config.reloc_ssim_thresh \
                and len(self.keyframe_manager.keyframes) >= self.config.warmup_config.relocalize_warmup:
                print('Relocalizing')

                # Set the relocalization state and pause densification.
                self.slam_state.relocalizing = True
                self.slam_state.densifying = False

                # Revert to before the last round of mapping.
                self.gaussian_manager.load_copy(gaussians_copy)
                self.trajectory_manager.load_copy(trajectory_copy)
                self.renderer.load_copy(renderer_copy)
                self.slam_state.global_mapping_step = global_mapping_step_copy
                self.keyframe_manager.update_visible_gaussians()
            else:
                if self.slam_state.relocalizing:
                    print('Relocalization complete.')
                self.slam_state.relocalizing = False
                self.slam_state.densifying = True

            if self.config.result_dir is not None:
                ssim_file_path = f'{self.config.result_dir}/SSIM.txt'
                if not os.path.exists(ssim_file_path):
                    with open(ssim_file_path, 'w') as f:
                        f.write('image index, kf index, SSIM, relocalizing\n')
                with open(ssim_file_path, 'a', encoding='utf8') as file:
                    file.write(f'{self.slam_state.current_index}, {len(self.keyframe_manager.keyframes) - 1}, '
                               f'{current_ssim}, {self.slam_state.relocalizing}\n')

    @NestedTimer.timed_function('Tracking')
    def track(
        self,
    ) -> bool:
        # Freeze the Gaussians, FPN offsets, and biases.
        self.gaussian_manager.freeze()
        self.renderer.freeze()
        self.trajectory_manager.freeze(biases=True)

        # Add control points.
        init_time = self.previous_timestamp.item()
        if self.slam_state.modeling_motion_artifacts:
            next_latest_time_eval = self.current_timestamp.item() + self.renderer.time_offsets.max().item()
        else:
            next_latest_time_eval = self.current_timestamp.item()
        if self.trajectory_manager.imu_frame:
            self.trajectory_manager.extend_splines(
                next_latest_time_eval,
                self.slam_state.predicting_poses,
                init_time,
                self.linear_accels_meas_imu,
                self.angular_vels_meas_imu,
                self.timestamps_imu)
        else:
            self.trajectory_manager.extend_splines(
                next_latest_time_eval,
                self.slam_state.predicting_poses,
                init_time)

        # Select recent IMU data.
        if self.dataset.imu_available:
            if self.slam_state.modeling_motion_artifacts:
                imu_start_time = torch.minimum(
                    self.previous_timestamp,
                    self.current_timestamp + self.renderer.time_offsets.min())
                imu_end_time = self.current_timestamp + self.renderer.time_offsets.max()
            else:
                imu_start_time = self.trajectory_manager.latest_time_eval
                imu_end_time = self.current_timestamp
            imu_selection_mask = (self.timestamps_imu >= imu_start_time) & (self.timestamps_imu < imu_end_time)
            selected_linear_accels = self.linear_accels_meas_imu[imu_selection_mask]
            selected_angular_vels = self.angular_vels_meas_imu[imu_selection_mask]
            selected_imu_timestamps = self.timestamps_imu[imu_selection_mask]

        # Perform tracking.
        for step in range(self.config.tracking_config.tracking_max_num_iter):
            with NestedTimer('Tracking Iteration'):
                self.viewer.pre_loop()

                # Zero the gradients.
                with NestedTimer('Zero Gradients'):
                    self.trajectory_manager.zero_grad()

                # Render the images.
                output_images, _, positions_camera_in_world, rotations_camera_to_world = self.renderer.render(
                    self.current_timestamp,
                    self.trajectory_manager,
                    self.slam_state.modeling_motion_artifacts,
                    mode='tracking')

                # Check for convergence.
                with NestedTimer('Check Convergence'):
                    if step != 0:
                        with torch.no_grad():
                            position_delta = (positions_camera_in_world - last_iter_positions).norm(dim=-1).mean()
                            rotation_delta = \
                                (rotations_camera_to_world.inv() * last_iter_rotations).log().norm(dim=-1).mean()
                        if position_delta < self.config.tracking_config.convergence_position_thresh and \
                            rotation_delta < self.config.tracking_config.convergence_rotation_thresh:
                            break
                    last_iter_positions = positions_camera_in_world
                    last_iter_rotations = rotations_camera_to_world

                with NestedTimer('Loss Computation'):
                    with NestedTimer('Radiometric Loss'):
                        # Compute the radiometric loss.
                        loss = torch.nn.functional.l1_loss(output_images['rendered_images'], self.current_image)

                    # Compute the IMU loss.
                    if self.dataset.imu_available:
                        loss += self.trajectory_manager.get_loss(selected_linear_accels, selected_angular_vels,
                            selected_imu_timestamps, use_reloc_weights=self.slam_state.relocalizing)

                # Compute the gradients.
                with NestedTimer('Backward Pass'):
                    loss.backward()

                # Update the parameters.
                with NestedTimer('Optimizer Updates'):
                    self.trajectory_manager.step()

                # Update the viewer.
                self.viewer.post_loop()

        # Update the viewer with the current image, current render, and the updated trajectory.
        self.viewer.update_frame(self.current_image)
        self.viewer.update_render(output_images['rendered_images'][-1])
        self.viewer.update_viewer_kfs(mode='tracking', tracking_timestamp=self.current_timestamp)

        # Unfreeze the Gaussians, FPN offsets, and biases.
        self.gaussian_manager.unfreeze()
        self.renderer.unfreeze()
        self.trajectory_manager.unfreeze(biases=True)

        # Perform keyframe selection.
        selected_as_keyframe = self.keyframe_manager.keyframe_selection(
            self.slam_state.current_index, self.current_image, self.current_timestamp)

        return selected_as_keyframe

    @NestedTimer.timed_function('IMU Initialization')
    def imu_initialization(
        self,
    ) -> None:
        # Run analytical IMU initialization.
        with NestedTimer('Analytical IMU Initialization'), torch.no_grad():
            print('Running analytical IMU initialization...')
            imu_init_kf_timestamps = [self.keyframe_manager.keyframes[0].timestamp]
            for kf in self.keyframe_manager.keyframes:
                if (kf.timestamp - imu_init_kf_timestamps[-1]) > (1 / self.config.imu_init_config.imu_update_pose_rate):
                    imu_init_kf_timestamps.append(kf.timestamp)
            imu_init_kf_timestamps = torch.cat(imu_init_kf_timestamps)
            transformations_camera_to_world = self.trajectory_manager(imu_init_kf_timestamps)
            bias_gyro, scale_factor, bias_accel, gravity_dir_world, _ = analytical_imu_initialization(
                transformations_camera_to_world=transformations_camera_to_world,
                timestamps_camera=imu_init_kf_timestamps,
                transformation_imu_to_camera=self.transformation_imu_to_camera,
                linear_accels_meas_imu=self.linear_accels_meas_imu,
                angular_vels_meas_imu=self.angular_vels_meas_imu,
                timestamps_imu=self.timestamps_imu,
                accelerometer_noise_density=self.dataset.accel_noise_density,
                gyroscope_noise_density=self.dataset.gyro_noise_density)

        # Update the trajectory.
        first_imu_update = not self.trajectory_manager.imu_frame
        self.trajectory_manager.update_imu_parameters(scale_factor, gravity_dir_world, bias_accel, bias_gyro)

        # Freeze the scale factor and gravity direction.
        self.trajectory_manager.freeze(gravity_dir=True, scale_factor=True)

        if first_imu_update:
            # Run mapping.
            print('Running mapping stage of IMU initialization...')
            self.map('imu_init', self.config.imu_init_config.imu_init_second_stage_num_iter)
            print('IMU initialization complete.')
        else:
            print('IMU update complete.')

        # Save the analytical IMU initialization/update results.
        if self.config.result_dir is not None:
            imu_update_dict = {
                'bias_gyro': bias_gyro.detach().tolist(),
                'bias_accel': bias_accel.detach().tolist(),
                'scale_factor': scale_factor,
                'gravity_dir_world': gravity_dir_world.detach().tolist()}
            os.makedirs(f'{self.config.result_dir}/imu_update', exist_ok=True)
            filepath = f'{self.config.result_dir}/imu_update/imu_update_{self.slam_state.current_index:06}.yaml'
            print(f'Saving the analytical IMU initialization/update results to: {filepath}')
            with open(filepath, 'w', encoding='utf8') as file:
                yaml.dump(imu_update_dict, file, default_flow_style=False, sort_keys=False)

    def save(
        self,
        suffix_extension: Optional[str] = None,
        save_timing: bool = False,
    ) -> None:
        # Compute ATE and update the ground truth positions in the viewer.
        ate_calculator = ATECalculator(self.dataset, self.trajectory_manager, self.keyframe_manager)
        if ate_calculator.enabled:
            print('Computing ATE (using keyframes and Sim3 alignment) and updating ground truth positions in the '
                  'viewer...')
            rmse_ate, percent_scale_error = ate_calculator.compute_rmse_ate(
                alignment='sim3',
                keyframes_only=True,
                current_index=self.slam_state.current_index,
                viewer=self.viewer)
            print(f'RMSE ATE: {rmse_ate}')
            if percent_scale_error is not None:
                print(f'Percent scale error: {percent_scale_error}')

        if self.config.result_dir is None:
            return

        if suffix_extension is not None:
            suffix = f'{self.slam_state.current_index:06}_{suffix_extension}'
        else:
            suffix = f'{self.slam_state.current_index:06}'
        print(f'Saving checkpoint with suffix: {suffix}')

        # Save the Gaussians, FPN, trajectory, and keyframes.
        self.gaussian_manager.save(self.config.result_dir, suffix)
        self.renderer.save(self.config.result_dir, suffix)
        self.trajectory_manager.save(self.config.result_dir, suffix)
        self.keyframe_manager.save(self.config.result_dir, suffix)

        # Save the SLAM state.
        os.makedirs(f'{self.config.result_dir}/slam_state', exist_ok=True)
        with open(f'{self.config.result_dir}/slam_state/slam_state_{suffix}.yaml', 'w', encoding='utf8') as file:
            yaml.dump(asdict(self.slam_state), file)

        # Save the maximum amount of GPU memory allocated since the last time a checkpoint was saved.
        mem_file_path = f'{self.config.result_dir}/max_memory_allocated.txt'
        if not os.path.exists(mem_file_path):
            with open(mem_file_path, 'w') as f:
                f.write('image index, max memory allocated (MiB)\n')

        max_memory_allocated = torch.cuda.max_memory_allocated(self.gaussian_device) / 1024**2 # Convert to MiB
        with open(mem_file_path, 'a', encoding='utf8') as file:
            file.write(f'{self.slam_state.current_index}, {max_memory_allocated}\n')
        torch.cuda.reset_peak_memory_stats(self.gaussian_device)

        # Save the current ATE.
        if ate_calculator.enabled:
            ate_file_path = f'{self.config.result_dir}/ATE.txt'
            if not os.path.exists(ate_file_path):
                with open(ate_file_path, 'w') as f:
                    f.write('image index, RMSE ATE, percent scale error\n')

            if percent_scale_error is None:
                percent_scale_error = -1.0
            with open(ate_file_path, 'a', encoding='utf8') as file:
                file.write(f'{self.slam_state.current_index}, {rmse_ate}, {percent_scale_error}\n')

        # Save the timing info.
        if save_timing and self.config.timer_config.enable:
            print('Saving timing data, this may take a while...')
            NestedTimer.save_time_data(
                output_folder=self.config.result_dir,
                save_timing_data=True,
                save_timing_summary=True,
                timing_data_filename=f'time_data.h5',
                timing_summary_filename=f'time_summary.yaml')

    def load(
        self,
    ) -> None:
        # Load the Gaussians, FPN, trajectory, and keyframes.
        self.gaussian_manager.load(self.config.load_dir, self.config.load_suffix, self.config.load_strategy_state,
            self.config.load_optimizer_states, self.config.load_freeze_states)
        self.renderer.load(self.config.load_dir, self.config.load_suffix, self.config.load_optimizer_states,
            self.config.load_freeze_states)
        self.trajectory_manager.load(self.config.load_dir, self.config.load_suffix, self.config.load_optimizer_states,
            self.config.load_freeze_states)
        self.keyframe_manager.load(self.config.load_dir, self.config.load_suffix)

        # Load the SLAM state.
        filepath_slam_state = f'{self.config.load_dir}/slam_state/slam_state_{self.config.load_suffix}.yaml'
        with open(filepath_slam_state, 'r') as file:
            saved_slam_state_dict = yaml.safe_load(file)
        self.slam_state = SLAMState(**saved_slam_state_dict)

        # Load the current image and previous timestamp.
        data_previous = self.dataset[self.slam_state.current_index - 1]
        self.previous_timestamp = data_previous['image_timestamp'].to(self.trajectory_device)
        data_current = self.dataset[self.slam_state.current_index]
        self.current_image = data_current['image'].to(self.gaussian_device)
        self.current_timestamp = data_current['image_timestamp'].to(self.trajectory_device)

        # Load IMU data up to the current image.
        if self.dataset.imu_available:
            selected_imu_data = self.dataset.get_imu_data_between_images(-1, self.slam_state.current_index)
            self.linear_accels_meas_imu = torch.from_numpy(selected_imu_data['linear_accels']).to(
                dtype=torch.double, device=self.trajectory_device)
            self.angular_vels_meas_imu = torch.from_numpy(selected_imu_data['angular_vels']).to(
                dtype=torch.double, device=self.trajectory_device)
            self.timestamps_imu = torch.from_numpy(selected_imu_data['timestamps']).to(
                self.trajectory_device)

        # Reinitialize the data loader iterator to start from the next image.
        remaining_indices = torch.arange(self.slam_state.current_index + 1, len(self.dataset))
        data_loader = torch.utils.data.DataLoader(self.dataset, sampler=remaining_indices)
        self.data_loader_iterator = iter(data_loader)

        # Update which Gaussians are visible to the current keyframe.
        self.keyframe_manager.update_visible_gaussians()

        # Compute ATE and update the ground truth positions in the viewer.
        ate_calculator = ATECalculator(self.dataset, self.trajectory_manager, self.keyframe_manager)
        if ate_calculator.enabled:
            ate_calculator.compute_rmse_ate(
                alignment='sim3',
                keyframes_only=True,
                current_index=self.slam_state.current_index,
                viewer=self.viewer)
