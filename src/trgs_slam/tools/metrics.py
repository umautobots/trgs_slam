from typing import Tuple, List, Optional, Literal
from pathlib import Path
import os

import h5py
import numpy as np
import cv2
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from scipy.spatial.transform import Rotation

from evo.tools import log
log.configure_logging(verbose=False, debug=False, silent=True)
from evo.core.trajectory import PoseTrajectory3D
from evo.core import metrics
from evo.core import lie_algebra

from trgs_slam.datasets.trnerf_dataset import TRNeRFDataset
from trgs_slam.renderer.renderer import Renderer
from trgs_slam.trajectory.trajectory import TrajectoryManager
from trgs_slam.trajectory.keyframes import KeyframeManager
from trgs_slam.tools.viewer import SLAMViewer

class ATECalculator:
    def __init__(
        self,
        dataset: TRNeRFDataset,
        trajectory_manager: TrajectoryManager,
        keyframe_manager: KeyframeManager,
    ) -> None:
        self.dataset = dataset
        self.trajectory_manager = trajectory_manager
        self.keyframe_manager = keyframe_manager

        self.enabled = self.dataset.ground_truth_poses is not None

    def compute_rmse_ate(
        self,
        alignment: Literal['sim3', 'se3', 'auto'] = 'auto',
        keyframes_only: bool = True,
        current_index: Optional[int] = None,
        viewer: Optional[SLAMViewer] = None,
        ground_truth_color: Tuple[int, int, int] = (255, 0, 0),
        ground_truth_line_width: float = 3,
        show_gt_traj_init_val: bool = False,
    ) -> Tuple[Optional[float], Optional[float]]:
        if not self.enabled:
            return None, None

        # Set the evaluation timestamps.
        if keyframes_only:
            timestamps = torch.cat([kf.timestamp for kf in self.keyframe_manager.keyframes]).cpu().numpy()
        else:
            assert current_index is not None, 'The current index is needed to evaluate the ATE at all images.'
            timestamps = self.dataset.image_timestamps[:current_index + 1]

        # Read in the ground_truth poses.
        if keyframes_only:
            poses_camera_to_world_gt = np.array(
                [self.dataset.ground_truth_poses[kf.index] for kf in self.keyframe_manager.keyframes])
        else:
            poses_camera_to_world_gt = self.dataset.ground_truth_poses[:current_index + 1]
        quats_gt = Rotation.from_matrix(poses_camera_to_world_gt[:, :3, :3].copy()).as_quat(scalar_first=True)
        pt_gt = PoseTrajectory3D(poses_camera_to_world_gt[:, :3, 3].copy(), quats_gt, timestamps)

        # Evaluate the estimated poses.
        with torch.no_grad():
            poses_camera_to_world_est = self.trajectory_manager(
                torch.from_numpy(timestamps).double()).detach().cpu().numpy()
        if self.trajectory_manager.imu_frame:
            poses_camera_to_world_est[:, :3, 3] *= self.trajectory_manager.trajectory.scale_factor.item()
        quats_est = Rotation.from_matrix(poses_camera_to_world_est[:, :3, :3].copy()).as_quat(scalar_first=True)
        pt_est = PoseTrajectory3D(poses_camera_to_world_est[:, :3, 3].copy(), quats_est, timestamps)

        # Compute ATE.
        correct_scale = True
        if alignment == 'se3' or (alignment == 'auto' and self.trajectory_manager.imu_frame):
            correct_scale = False
        align_rotation, align_translation, align_scale = pt_est.align(pt_gt, correct_scale=correct_scale)
        ate_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ate_metric.process_data((pt_gt, pt_est))
        rmse_ate = ate_metric.get_statistic(metrics.StatisticsType.rmse)

        # Compute the percent scale error.
        percent_scale_error = None
        if self.trajectory_manager.imu_frame:
            if correct_scale:
                percent_scale_error = np.abs(align_scale - 1.0) * 100.0
            else:
                # Perform alignment again with scale correction.
                quats_est = \
                    Rotation.from_matrix(poses_camera_to_world_est[:, :3, :3].copy()).as_quat(scalar_first=True)
                pt_est = PoseTrajectory3D(poses_camera_to_world_est[:, :3, 3].copy(), quats_est, timestamps)
                _, _, align_scale_sim3 = pt_est.align(pt_gt, correct_scale=True)
                percent_scale_error = np.abs(align_scale_sim3 - 1.0) * 100.0

        # Add the ground truth trajectory to the viewer.
        if viewer is not None and not viewer.config.disable_viewer:
            # Objects in the viewer have relative scale and are aligned with the SLAM world frame. Therefore, we need to
            # invert the transformation that aligned the estimated trajectory with the ground truth, and rescale
            # the ground truth.
            align_inverse = lie_algebra.se3_inverse(lie_algebra.se3(align_rotation, align_translation))
            pt_gt.transform(align_inverse)
            pt_gt.scale(1.0 / align_scale)
            if self.trajectory_manager.imu_frame:
                pt_gt.scale(1.0 / self.trajectory_manager.trajectory.scale_factor.item())

            # Add the ground truth line segment to the viewer.
            gt_traj_handle = viewer.server.scene.get_handle_by_name('gt_line_segment')
            visiblity = gt_traj_handle.visible if gt_traj_handle is not None else show_gt_traj_init_val
            viewer.server.scene.add_line_segments(
                name='gt_line_segment',
                points=np.stack([pt_gt.positions_xyz[:-1], pt_gt.positions_xyz[1:]], axis=1),
                colors=ground_truth_color,
                line_width=ground_truth_line_width,
                visible=visiblity)

            # Add check box for showing the ground truth trajectory.
            with viewer._rendering_folder:
                if not 'show_gt_traj_checkbox' in viewer._rendering_tab_handles:
                    show_gt_traj_checkbox = viewer.server.gui.add_checkbox(
                        'Show GT Positions',
                        initial_value=show_gt_traj_init_val,
                        hint='Show the ground truth trajectory.')
                    viewer._rendering_tab_handles['show_gt_traj_checkbox'] = show_gt_traj_checkbox

                    @show_gt_traj_checkbox.on_update
                    def _(_) -> None:
                        gt_traj_handle = viewer.server.scene.get_handle_by_name('gt_line_segment')
                        gt_traj_handle.visible = show_gt_traj_checkbox.value

        return rmse_ate, percent_scale_error

class TRNeRFLPIPSCalculator:
    def __init__(
        self,
        dataset: TRNeRFDataset,
        renderer: Renderer,
        trajectory_manager: TrajectoryManager,
    ) -> None:
        self.dataset = dataset
        self.renderer = renderer
        self.trajectory_manager = trajectory_manager

        sequence = Path(self.dataset.config.path_images).parent.name
        self.enabled = False
        if 'slow' in sequence or self.dataset.config.path_ground_truth_poses is None:
            return

        # Infer the paths to the pseudo ground truth images and LPIPS timestamps.
        sequence_folder = str(Path(self.dataset.config.path_images).parent)
        path_ground_truth_images = \
            f"{sequence_folder}/pseudo_ground_truth_{self.dataset.image_data_dict['name']}.h5"
        sequence_folder = str(Path(self.dataset.config.path_ground_truth_poses).parent)
        path_lpips_timestamps = f"{sequence_folder}/lpips_timestamps.txt"
        if not (os.path.exists(path_ground_truth_images) and os.path.exists(path_lpips_timestamps)):
            return

        # Open H5 ground truth image file.
        h5_gt_image_file = h5py.File(path_ground_truth_images, 'r')
        group_gt_image = h5_gt_image_file['image_rendered']
        self.gt_image_data_dict = {
            'images': group_gt_image['images'],
            'timestamps': group_gt_image['timestamps'][:]}

        # Open the LPIPS timestamps file.
        self.lpips_timestamps = np.loadtxt(path_lpips_timestamps, dtype=np.uint64)

        # Set the thermal image thresholds to use in evaluation. These are hardcoded to match those used in the TRNeRF
        # paper.
        if 'indoor' in sequence:
            if 'left' in self.dataset.image_data_dict['name']:
                self.threshold_min=22848
                self.threshold_max=23468
            else:
                self.threshold_min=22884
                self.threshold_max=23486
        else:
            if 'left' in self.dataset.image_data_dict['name']:
                self.threshold_min=22476
                self.threshold_max=25473
            else:
                self.threshold_min=22460
                self.threshold_max=25598

        # Compute the full resolution new camera matrix.
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.dataset.original_camera_matrix_full_res,
            self.dataset.distortion_coefficients,
            (self.dataset.image_width_full_res, self.dataset.image_height_full_res),
            alpha=0)
        self.new_camera_matrix = torch.from_numpy(new_camera_matrix).to(
            dtype=torch.float, device=self.renderer.gaussian_device)

        # Initialize the LPIPS calculator.
        self.lpips_calculator = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.renderer.gaussian_device)

        self.enabled = True

    def compute_lpips(
        self,
    ) -> Tuple[Optional[List[float]], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        if not self.enabled:
            return None, None, None

        lpips_results = []
        gt_images = []
        restored_images = []

        for timestamp in self.lpips_timestamps:
            if timestamp * 1e-9  < self.trajectory_manager.time_start or \
                timestamp * 1e-9 > self.trajectory_manager.latest_time_eval:
                continue

            # Read in the pseudo ground truth image.
            index_image = np.argwhere(self.gt_image_data_dict['timestamps'] == timestamp)[0][0]

            # Rescale the pseudo ground truth image and convert it to 8-bit.
            gt_image = self.gt_image_data_dict['images'][index_image]
            gt_image[gt_image < self.threshold_min] = self.threshold_min
            gt_image[gt_image > self.threshold_max] = self.threshold_max
            gt_image = (gt_image.astype(np.float32) - self.threshold_min) / (self.threshold_max - self.threshold_min)
            gt_image = np.round(gt_image * 255.0).astype(np.uint8)
            gt_image = torch.from_numpy(gt_image).to(dtype=torch.float, device=self.renderer.gaussian_device)

            # Rasterize the restored image.
            timestamp = torch.tensor([timestamp * 1e-9],
                dtype=torch.double, device=self.trajectory_manager.config.device)
            with torch.no_grad():
                transformation_world_to_camera = self.trajectory_manager(timestamp, as_matrices=True, invert=True).to(
                    dtype=torch.float, device=self.renderer.gaussian_device)
                rasterized_image, _, _ = self.renderer.rasterize(
                    transformations_world_to_camera=transformation_world_to_camera,
                    Ks=self.new_camera_matrix.unsqueeze(0),
                    width=self.dataset.image_width_full_res,
                    height=self.dataset.image_height_full_res)
            rasterized_image = rasterized_image.squeeze()

            # Rescale the restored image to match the thresholds used in TRNeRF's LPIPS evaluation and convert it to
            # 8-bit.
            rasterized_image = rasterized_image * (self.dataset.threshold_maximum - self.dataset.threshold_minimum) + \
                self.dataset.threshold_minimum
            rasterized_image[rasterized_image < self.threshold_min] = self.threshold_min
            rasterized_image[rasterized_image > self.threshold_max] = self.threshold_max
            rasterized_image = (rasterized_image - self.threshold_min) / (self.threshold_max - self.threshold_min)
            rasterized_image = (rasterized_image * 255.0).round().to(torch.uint8).float()

            # For the LPIPS calculator, the images need to have shape (N, 3, H, W) and be in the range [0, 1]
            gt_image = torch.unsqueeze(torch.unsqueeze(gt_image, 0), 0)
            gt_image = gt_image.repeat(1, 3, 1, 1) / 255.0
            rasterized_image = torch.unsqueeze(torch.unsqueeze(rasterized_image, 0), 0)
            rasterized_image = rasterized_image.repeat(1, 3, 1, 1) / 255.0

            # Compute the LPIPS metric.
            lpips = self.lpips_calculator(rasterized_image, gt_image).item()

            # Store the results.
            lpips_results.append(lpips)
            gt_images.append(gt_image[0, 0, :, :])
            restored_images.append(rasterized_image[0, 0, :, :])

        return lpips_results, gt_images, restored_images
