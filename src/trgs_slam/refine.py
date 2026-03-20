from dataclasses import dataclass, field

import yaml
import numpy as np
import torch
import os
import cv2

from trgs_slam.slam import SLAMConfig, SLAM
from trgs_slam.tools.nested_timer import NestedTimer
from trgs_slam.datasets.trnerf_dataset import TRNeRFDataset
from trgs_slam.tools.metrics import ATECalculator, TRNeRFLPIPSCalculator

@dataclass
class RefineConfig:
    # Number of optimization steps for refinement.
    refine_num_iter: int = 15_000
    # Batch size of images to sample at each iteration.
    batch_size: int = 10
    # Whether to freeze the gravity direction.
    freeze_gravity_dir: bool = False
    # Whether to freeze the scale factor.
    freeze_scale: bool = True
    # Whether to use all IMU data at each iteration.
    use_all_imu_data: bool = True
    # The interval of time (in seconds), centered on each sampled image, within which to sample IMU data. Ignored if
    # use_all_imu_data is True.
    imu_interval: float = 0.2
    # An exponentially decaying learning rate scheduler is used with the Gaussian means. This argument sets the
    # multiplier for the final learning rate relative to the initial learning rate.
    means_lr_final_factor: float = 0.001
    # How frequently to compute metrics (in terms of optimization steps). Metrics will also be computed after the final
    # step.
    compute_metrics_interval: int = 500
    # Save the rendered and pseudo ground truth images used in the LPIPS calculation.
    save_lpips_images: bool = True
    # SLAM options.
    slam_config: SLAMConfig = field(default_factory=SLAMConfig)

class Refine(SLAM):
    def __init__(
        self,
        config: RefineConfig
    ) -> None:
        self.refine_config = config

        # Load the SLAM result.
        self.refine_config.slam_config.keyframe_config.cache_location = 'disk' # Skip loading keyframe images.
        super().__init__(self.refine_config.slam_config)
        self.keyframe_manager.window_indices = [] # Clear the keyframe window -- not relevant outside of SLAM.

        # Save the config.
        if self.config.result_dir is not None:
            with open(f'{self.config.result_dir}/refine_config.yaml', 'w', encoding='utf8') as file:
                yaml.dump(self.refine_config, file)

        # Create a collate function that drops the IMU data. The IMU data was already read in when the SLAM result was
        # loaded and it interferes with batching.
        def collate_fn_no_imu(batch):
            keys_to_drop = ['linear_accels', 'angular_vels', 'imu_timestamps']
            filtered_batch = []
            for sample in batch:
                filtered_sample = {k: v for k, v in sample.items() if k not in keys_to_drop}
                filtered_batch.append(filtered_sample)
            return torch.utils.data.default_collate(filtered_batch)

        # Initialize the data loader.
        sampler = torch.utils.data.SubsetRandomSampler(torch.arange(self.slam_state.current_index + 1))
        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.refine_config.batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=collate_fn_no_imu,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True)
        data_loader_iterator = iter(data_loader)

        # Extend the spline if needed.
        if self.config.renderer_config.num_rasters > 1:
            next_latest_time_eval = (self.current_timestamp + self.renderer.time_offsets.max()).item()
            self.trajectory_manager.extend_splines(next_latest_time_eval, predict_poses=False, enable_timing=False)

        # Freeze selected states.
        self.trajectory_manager.freeze(
            gravity_dir=self.refine_config.freeze_gravity_dir,
            scale_factor=self.refine_config.freeze_scale)

        # Set the Gaussian mean learning rate scheduler.
        means_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.gaussian_manager.optimizers['means'],
            gamma=self.refine_config.means_lr_final_factor**(1.0 / self.refine_config.refine_num_iter))

        # Setup the ATE calculator.
        self.ate_calculator = ATECalculator(self.dataset, self.trajectory_manager, self.keyframe_manager)

        # If the dataset is TRNeRF, set up the LPIPS calculator.
        if type(self.dataset) is TRNeRFDataset:
            self.trnerf_lpips_calculator = TRNeRFLPIPSCalculator(self.dataset, self.renderer, self.trajectory_manager)

        # Run refinement.
        for step in range(self.refine_config.refine_num_iter):
            # Compute metrics.
            if step % self.refine_config.compute_metrics_interval == 0:
                self.compute_metrics(step)

            # Load data.
            try:
                data = next(data_loader_iterator)
            except StopIteration:
                data_loader_iterator = iter(data_loader)
                data = next(data_loader_iterator)
            images = data['image'].to(self.gaussian_device)
            timestamps = data['image_timestamp'].to(self.trajectory_device).squeeze()

            with NestedTimer('Refinement Iteration'):
                self.viewer.pre_loop()

                # Zero the gradients.
                self.gaussian_manager.zero_grad()
                self.renderer.zero_grad()
                self.trajectory_manager.zero_grad()

                # Select IMU data.
                if self.dataset.imu_available:
                    if self.refine_config.use_all_imu_data:
                        imu_selection_mask = self.timestamps_imu <= self.trajectory_manager.latest_time_eval
                        selected_linear_accels = self.linear_accels_meas_imu[imu_selection_mask]
                        selected_angular_vels = self.angular_vels_meas_imu[imu_selection_mask]
                        selected_imu_timestamps = self.timestamps_imu[imu_selection_mask]
                    else:
                        imu_start_times = timestamps - self.refine_config.imu_interval / 2.0
                        imu_end_times = timestamps + self.refine_config.imu_interval / 2.0
                        imu_end_times[imu_end_times > self.trajectory_manager.latest_time_eval] = \
                            self.trajectory_manager.latest_time_eval
                        imu_selection_mask = \
                            ((self.timestamps_imu >= imu_start_times.unsqueeze(-1)) &
                            (self.timestamps_imu <= imu_end_times.unsqueeze(-1))).any(dim=0)
                        selected_linear_accels = self.linear_accels_meas_imu[imu_selection_mask]
                        selected_angular_vels = self.angular_vels_meas_imu[imu_selection_mask]
                        selected_imu_timestamps = self.timestamps_imu[imu_selection_mask]

                # Render the images.
                output_images, info, _, _ = self.renderer.render(
                    timestamps,
                    self.trajectory_manager,
                    model_motion_artifacts=True)

                # Retain necessary gradients for the densification strategy.
                self.gaussian_manager.strategy_step_pre_backward(step, info)

                with NestedTimer('Loss Computation'):
                    # Compute the radiometric loss.
                    loss = torch.nn.functional.l1_loss(output_images['rendered_images'], images)

                    # Compute the FPN pixelwise offsets loss.
                    loss += self.renderer.get_loss(timestamps)

                    # Compute the IMU loss with the selected IMU measurements.
                    if self.dataset.imu_available:
                        loss += self.trajectory_manager.get_loss(
                            selected_linear_accels,
                            selected_angular_vels,
                            selected_imu_timestamps)

                with NestedTimer('Backward Pass'):
                    # Compute the gradients.
                    loss.backward()

                with NestedTimer('Optimizer Updates'):
                    # Update the parameters.
                    self.gaussian_manager.step(info)
                    self.renderer.step()
                    self.trajectory_manager.step()
                    means_lr_scheduler.step()

                # Accumulate information or perform densification and pruning.
                self.gaussian_manager.strategy_step_post_backward(step, info)

                # Update the viewer.
                self.viewer.post_loop(step)

        # Compute final metrics.
        self.compute_metrics(self.refine_config.refine_num_iter)

    def compute_metrics(
        self,
        step: int,
    ) -> None:
        if self.config.result_dir is not None:
            # Save the maximum amount of GPU memory allocated since the last time the metrics were computed.
            mem_file_path = f'{self.config.result_dir}/max_memory_allocated.txt'
            if not os.path.exists(mem_file_path):
                with open(mem_file_path, 'w') as f:
                    f.write('optimization step, max memory allocated (MiB)\n')

            max_memory_allocated = torch.cuda.max_memory_allocated(self.gaussian_device) / 1024**2 # Convert to MiB
            with open(mem_file_path, 'a', encoding='utf8') as file:
                file.write(f'{step}, {max_memory_allocated}\n')
            torch.cuda.reset_peak_memory_stats(self.gaussian_device)

        # Compute ATE and update the ground truth positions in the viewer.
        if self.ate_calculator.enabled:
            print('Computing ATE and updating ground truth positions in the viewer...')
            rmse_ate, percent_scale_error = self.ate_calculator.compute_rmse_ate(
                alignment='auto',
                keyframes_only=False,
                current_index=self.slam_state.current_index,
                viewer=self.viewer,
                show_gt_traj_init_val=True)
            print(f'RMSE ATE: {rmse_ate}')
            if percent_scale_error is not None:
                print(f'Percent scale error: {percent_scale_error}')

            if self.config.result_dir is not None:
                ate_file_path = f'{self.config.result_dir}/ATE.txt'
                if not os.path.exists(ate_file_path):
                    with open(ate_file_path, 'w') as f:
                        f.write('optimization step, RMSE ATE, percent scale error\n')

                if percent_scale_error is None:
                    percent_scale_error = -1.0
                with open(f'{self.config.result_dir}/ATE.txt', 'a', encoding='utf8') as file:
                    file.write(f'{step}, {rmse_ate}, {percent_scale_error}\n')

        # Compute LPIPS.
        if type(self.dataset) is TRNeRFDataset and self.trnerf_lpips_calculator.enabled:
            print('Computing LPIPS...')
            lpips_results, gt_images, restored_images = self.trnerf_lpips_calculator.compute_lpips()
            if len(lpips_results) != 0:
                print(f'Mean LPIPS: {np.mean(lpips_results)}')

                if self.config.result_dir is not None:
                    lpips_file_path = f'{self.config.result_dir}/LPIPS.txt'
                    if not os.path.exists(lpips_file_path):
                        with open(lpips_file_path, 'w') as f:
                            f.write('optimization step, mean LPIPS\n')

                    with open(lpips_file_path, 'a', encoding='utf8') as file:
                        file.write(f'{step}, {np.mean(lpips_results)}\n')

                    if self.refine_config.save_lpips_images:
                        os.makedirs(f'{self.config.result_dir}/lpips_images/rendered/step_{step:06}/', exist_ok=True)
                        os.makedirs(f'{self.config.result_dir}/lpips_images/gt/', exist_ok=True)
                        for i in range(len(gt_images)):
                            gt_image = gt_images[i].detach().clone().cpu().numpy()
                            gt_image = (gt_image * 255).astype(np.uint8)
                            cv2.imwrite(
                                f'{self.config.result_dir}/lpips_images/gt/{i:04}.png',
                                gt_image)
                        for i in range(len(restored_images)):
                            restored_image = restored_images[i].detach().clone().cpu().numpy()
                            restored_image = (restored_image * 255).astype(np.uint8)
                            cv2.imwrite(
                                f'{self.config.result_dir}/lpips_images/rendered/step_{step:06}/{i:04}.png',
                                restored_image)
            else:
                print('Skipped. No pseudo ground truth images are in the splines\' time range.')
