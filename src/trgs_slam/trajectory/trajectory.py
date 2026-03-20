from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import os
import math
import copy

import torch

import pypose as pp
import lietorch as lt

from lie_spline_torch.uniform_rd_bspline import UniformRdBSpline
from lie_spline_torch.uniform_so3_bspline import UniformSO3BSpline

from trgs_slam.trajectory.imu_utils import integrate_imu, GRAVITY_MAGNITUDE
from trgs_slam.tools.nested_timer import NestedTimer
from trgs_slam.datasets.base_dataset import BaseDataset

def to_transformation_matrices(
    positions: torch.Tensor,
    rotations: lt.SO3,
    invert: bool = False,
) -> torch.Tensor:
    if invert:
        rotations_inverse = rotations.inv()
        transformations = rotations_inverse.matrix()
        transformations[:, :3, 3] = -(rotations_inverse * positions)
    else:
        transformations = rotations.matrix()
        transformations[:, :3, 3] = positions

    return transformations

@dataclass
class TrajectoryConfig:
    # Device on which to store the trajectory control points and perform trajectory based computations (e.g., computing
    # the IMU based loss).
    device: str = 'cpu'
    # Whether to evaluate the rotation spline on the CPU using the C++ implementation. This can be enabled regardless
    # of the device the trajectory control points are stored on.
    use_cpp: bool = True
    # Order of the position and rotation splines. The default, 4, ensures C2 continuity (i.e., the second derivatives
    # are continuous).
    pose_spline_order: int = 4
    # Knot time interval for the position and rotation splines (in seconds). If <= 0.0, it will be set to half the
    # average frame period.
    pose_knot_interval: float = -1.0
    # Learning rate for the position control points.
    position_lr: float = 1e-3
    # Learning rate for the rotation control points.
    rotation_lr: float = 1e-4
    # Learning rate for the IMU biases.
    bias_lr: float = 1e-3
    # Learning rate for the gravity direction. Ignored in SLAM.
    gravity_dir_lr: float = 1e-4
    # Learning rate for the scale factor. Ignored in SLAM.
    scale_lr: float = 1e-4
    # Weight for the linear acceleration loss.
    linear_accel_weight: float = 1e-4
    # Weight for the angular velocity loss.
    angular_vel_weight: float = 1e-2
    # Weight for the linear acceleration loss during relocalization.
    linear_accel_weight_reloc: float = 5e-6
    # Weight for the angular velocity loss during relocalization.
    angular_vel_weight_reloc: float = 5e-4
    # Weight for the accelerometer bias loss.
    bias_accel_weight: float = 1e-1
    # Weight for the gyroscope bias loss.
    bias_gyro_weight: float = 1e-1
    # The number of predicted poses to use per knot in spline fitting. When the position and rotation splines are
    # extended it is done in one of three ways: duplication, constant velocity, or IMU integration. In the first method
    # the most recent position and rotation control points are simply duplicated. In the other two methods, future poses
    # are predicted, either through a constant velocity model or through IMU integration, and the spline is fit to these
    # predictions. This argument determines how many predicted poses are used for this as a multiple of the number of
    # spline knots across the time range of the predictions (which typically begins at the midpoint of the previous
    # frame and ends at the last raster timestamp for the current frame).
    num_fit_per_knot: int = 10
    # The size of the time interval (in seconds) used for linear and angular velocity estimation when predicting new
    # control points with a constant velocity model. If > 0.0, the velocities are estimated with the finite difference
    # method using this value as the interval size. If < 0.0, the interval is set to the average frame period.
    # If == 0.0, the instantaneous velocities (first derivatives of the position and rotation splines) are used.
    vel_avg_interval: float = -1.0

class Trajectory(torch.nn.Module):
    def __init__(
        self,
        pose_spline_order: int,
        pose_knot_interval: float,
        use_cpp: bool,
        time_start: float,
        imu_frame: bool = False,
        transformation_imu_to_camera: Optional[torch.Tensor] = None,
        position_control_points: Optional[torch.Tensor] = None,
        rotation_control_points: Optional[torch.Tensor] = None,
        bias_gyro: torch.Tensor = torch.zeros(3, dtype=torch.double),
        bias_accel: torch.Tensor = torch.zeros(3, dtype=torch.double),
        scale_factor: torch.Tensor = torch.ones(1, dtype=torch.double),
        gravity_dir_world: torch.Tensor = torch.tensor([0.0, 1.0, 0.0], dtype=torch.double),
    ) -> None:
        super().__init__()

        # Initialize the position and rotation splines
        if position_control_points is None or rotation_control_points is None:
            position_control_points = torch.zeros(pose_spline_order, 3)
            rotation_control_points = pp.identity_SO3(pose_spline_order).tensor()
        self.position_spline = UniformRdBSpline(pose_spline_order, pose_knot_interval, time_start,
            position_control_points.to(torch.double), sparse=True)
        self.rotation_spline = UniformSO3BSpline(pose_spline_order, pose_knot_interval, time_start,
            rotation_control_points.to(torch.double), sparse=True, use_cpp=use_cpp, use_pypose=False)

        # Setup IMU extrinsics and related parameters.
        assert not (imu_frame and transformation_imu_to_camera is None), 'The trajectory cannot represent the IMU ' \
            'frame without the camera-IMU extrinsics.'
        self.imu_frame = imu_frame
        if transformation_imu_to_camera is not None:
            # Store the extrinsics between the camera and the IMU
            transformation_camera_to_imu = transformation_imu_to_camera.inverse()
            self.rotation_camera_to_imu = lt.SO3(
                pp.from_matrix(transformation_camera_to_imu[:3, :3], ltype=pp.SO3_type).tensor().unsqueeze(0))
            self.position_camera_in_imu = transformation_camera_to_imu[:3, 3].unsqueeze(0)

            # Initialize the parameter for the gyroscope bias.
            self.bias_gyro = torch.nn.Parameter(bias_gyro)

            if self.imu_frame:
                # Initialize the parameter for the accelerometer bias.
                self.bias_accel = torch.nn.Parameter(bias_accel)

                # Initialize a scalar parameter to estimate the scale factor that takes the camera poses and Gaussians
                # to absolute scale.
                self.scale_factor = torch.nn.Parameter(scale_factor)

                # Initialize the parameter for the gravity direction in the world frame.
                self.gravity_dir_world = torch.nn.Parameter(gravity_dir_world)

    def forward(
        self,
        times_eval: torch.Tensor,
        as_matrices: bool = True,
        invert: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, lt.SO3]]:
        if self.imu_frame:
            # Evaluate the splines.
            positions_imu_in_world = self.position_spline(times_eval)
            rotations_imu_to_world = self.rotation_spline(times_eval)

            # Convert the poses from IMU->world to camera->world.
            positions_camera_in_world = rotations_imu_to_world * self.position_camera_in_imu + positions_imu_in_world
            rotations_camera_to_world = rotations_imu_to_world * self.rotation_camera_to_imu

            # Apply the scale factor to convert the camera positions from absolute scale to the relative scale
            # consistent with the Gaussians.
            positions_camera_in_world = positions_camera_in_world / self.scale_factor
        else:
            # Evaluate the splines.
            positions_camera_in_world = self.position_spline(times_eval)
            rotations_camera_to_world = self.rotation_spline(times_eval)

        if as_matrices:
            return to_transformation_matrices(positions_camera_in_world, rotations_camera_to_world, invert)
        else:
            if invert:
                rotations_world_to_camera = rotations_camera_to_world.inv()
                positions_world_in_camera = -(rotations_world_to_camera * positions_camera_in_world)
                return positions_world_in_camera, rotations_world_to_camera
            else:
                return positions_camera_in_world, rotations_camera_to_world

    def get_loss(
        self,
        linear_accels_meas_imu: torch.Tensor,
        angular_vels_meas_imu: torch.Tensor,
        timestamps_imu: torch.Tensor,
        latest_random_timestamp: Optional[torch.Tensor] = None,
        weight_random: Optional[float] = None,
        weight_window: Optional[float] = None,
    ) -> torch.Tensor:
        if self.imu_frame:
            # Remove the bias from the accelerometer measurement.
            unbiased_linear_accels_meas_imu = linear_accels_meas_imu - self.bias_accel

            # Evaluate the rotation of the IMU frame and rotate the accelerometer measurement from the IMU frame to the
            # world frame.
            rotations_imu_to_world = self.rotation_spline(timestamps_imu)
            linear_accels_meas_world = rotations_imu_to_world * unbiased_linear_accels_meas_imu

            # Remove gravity from the accelerometer measurement.
            gravity_world = (self.gravity_dir_world / self.gravity_dir_world.norm()) * GRAVITY_MAGNITUDE
            absolute_linear_accels_meas_world = linear_accels_meas_world + gravity_world

            # Evaluate the acceleration of the position spline (the absolute linear acceleration of the IMU expressed in
            # the world frame).
            absolute_linear_accels_est_world = self.position_spline.evaluate_accelerations(timestamps_imu)

            # Compute the linear acceleration error.
            error_linear_accels = (absolute_linear_accels_est_world - absolute_linear_accels_meas_world).norm(dim=-1)

            # Remove the bias from the gyroscope measurement.
            unbiased_angular_vels_meas_imu = angular_vels_meas_imu - self.bias_gyro

            # Evaluate the velocity of the rotation spline (the angular velocity of the IMU expressed in the IMU frame).
            angular_vels_est_imu = self.rotation_spline.evaluate_velocities(timestamps_imu)

            # Compute the angular velocity error.
            error_angular_vels = (angular_vels_est_imu - unbiased_angular_vels_meas_imu).norm(dim=-1)
        else:
            # Remove the bias from the gyroscope measurement.
            unbiased_angular_vels_meas_imu = angular_vels_meas_imu - self.bias_gyro

            # Evaluate the velocity of the rotation spline (the angular velocity of the camera expressed in the camera
            # frame).
            angular_vels_est_camera = self.rotation_spline.evaluate_velocities(timestamps_imu)

            # Rotate the angular velocity estimate from the camera frame to the IMU frame.
            angular_vels_est_imu = self.rotation_camera_to_imu * angular_vels_est_camera

            # Compute the angular velocity error.
            error_angular_vels = (angular_vels_est_imu - unbiased_angular_vels_meas_imu).norm(dim=-1)

        # Compute the linear acceleration and angular velocity losses.
        if not (latest_random_timestamp is None or weight_random is None or weight_window is None):
            # Weight the errors to maintain the correct balance between the radiometric loss and IMU losses.
            weight_mask_random = timestamps_imu <= latest_random_timestamp
            weights = torch.ones_like(timestamps_imu)
            weights[weight_mask_random] *= weight_random
            weights[~weight_mask_random] *= weight_window
            loss_linear_accels = (error_linear_accels * weights).mean() if self.imu_frame else 0.0
            loss_angular_vels = (error_angular_vels * weights).mean()
        else:
            loss_linear_accels = error_linear_accels.mean() if self.imu_frame else 0.0
            loss_angular_vels = error_angular_vels.mean()

        # Compute accelerometer and gyroscope bias losses.
        loss_bias_accel = self.bias_accel.norm() if self.imu_frame else 0.0
        loss_bias_gyro = self.bias_gyro.norm()

        return loss_linear_accels, loss_angular_vels, loss_bias_accel, loss_bias_gyro

class TrajectoryManager():
    def __init__(
        self,
        config: TrajectoryConfig,
        dataset: BaseDataset,
        time_start: float,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.time_start = time_start
        self.latest_time_eval = time_start

        # Set default intervals, if needed.
        self.config.pose_knot_interval = self.config.pose_knot_interval if self.config.pose_knot_interval > 0.0 else \
            self.dataset.average_frame_period / 2.0
        self.config.vel_avg_interval = self.config.vel_avg_interval if self.config.vel_avg_interval >= 0.0 else \
            self.dataset.average_frame_period

        # Initialize the trajectory and optimizers.
        if self.dataset.imu_available:
            self.transformation_imu_to_camera = torch.from_numpy(self.dataset.transformation_imu_to_camera).to(
                dtype=torch.double, device=self.config.device)
        else:
            self.transformation_imu_to_camera = None
        self.imu_frame = False
        self.initialize_trajectory()
        self.initialize_optimizers()

    def initialize_trajectory(
        self,
        **kwargs,
    ) -> None:
        self.trajectory = Trajectory(
            self.config.pose_spline_order,
            self.config.pose_knot_interval,
            self.config.use_cpp,
            self.time_start,
            self.imu_frame,
            self.transformation_imu_to_camera,
            **kwargs).to(self.config.device)

    def initialize_optimizers(
        self,
    ) -> None:
        self.position_optimizer = torch.optim.SparseAdam(self.trajectory.position_spline.parameters(),
            lr=self.config.position_lr)
        self.rotation_optimizer = torch.optim.SparseAdam(self.trajectory.rotation_spline.parameters(),
            lr=self.config.rotation_lr)
        if self.dataset.imu_available and not self.imu_frame:
            self.bias_optimizer = torch.optim.Adam([self.trajectory.bias_gyro], lr=self.config.bias_lr)
        if self.imu_frame:
            self.bias_optimizer = torch.optim.Adam([self.trajectory.bias_accel, self.trajectory.bias_gyro],
                lr=self.config.bias_lr)
            self.gravity_dir_optimizer = torch.optim.Adam([self.trajectory.gravity_dir_world],
                lr=self.config.gravity_dir_lr)
            self.scale_optimizer = torch.optim.Adam([self.trajectory.scale_factor],
                lr=self.config.scale_lr)

    @NestedTimer.timed_function('Spline Extension')
    @torch.no_grad()
    def extend_splines(
        self,
        next_latest_time_eval: float,
        predict_poses: bool,
        init_time: Optional[float] = None,
        linear_accels_meas_imu: Optional[torch.Tensor] = None,
        angular_vels_meas_imu: Optional[torch.Tensor] = None,
        timestamps_imu: Optional[torch.Tensor] = None,
    ) -> None:
        self.trajectory.position_spline.extend_to_time(next_latest_time_eval, self.position_optimizer)
        self.trajectory.rotation_spline.extend_to_time(next_latest_time_eval, self.rotation_optimizer)
        if not predict_poses:
            return

        init_time = self.latest_time_eval if init_time is None else init_time
        if not self.imu_frame:
            # Predict positions and rotations using a constant velocity model and fit the splines to them.
            self.trajectory.position_spline.fit_to_constant_velocity(init_time, next_latest_time_eval,
                self.config.vel_avg_interval, self.config.num_fit_per_knot)
            self.trajectory.rotation_spline.fit_to_constant_velocity(init_time, next_latest_time_eval,
                self.config.vel_avg_interval, self.config.num_fit_per_knot)
        else:
            # Set the timestamps for the poses to predict.
            init_time = torch.tensor([init_time], device=self.config.device, dtype=torch.double)
            num_fit = int(math.ceil((next_latest_time_eval - init_time) / self.config.pose_knot_interval)) * \
                self.config.num_fit_per_knot
            times_predict = torch.linspace(init_time[0], next_latest_time_eval, num_fit,
                dtype=torch.double, device=self.config.device)

            # Evaluate the splines at the init time.
            init_position = self.trajectory.position_spline(init_time)
            init_linear_velocity = self.trajectory.position_spline.evaluate_velocities(init_time)
            init_rotation = self.trajectory.rotation_spline(init_time)

            # Compute the predicted positions and rotations through IMU integration.
            predicted_positions, predicted_rotations = integrate_imu(
                times_predict,
                init_rotation,
                init_linear_velocity,
                init_position,
                init_time,
                linear_accels_meas_imu,
                angular_vels_meas_imu,
                timestamps_imu,
                self.trajectory.bias_accel,
                self.trajectory.bias_gyro,
                self.trajectory.gravity_dir_world)

            # Fit the splines to the predicted positions and rotations.
            self.trajectory.position_spline.fit_control_points(times_predict, predicted_positions)
            self.trajectory.rotation_spline.fit_control_points(times_predict, predicted_rotations)

    def update_imu_parameters(
        self,
        scale_factor: float,
        gravity_dir_world: torch.Tensor,
        bias_accel: torch.Tensor,
        bias_gyro: torch.Tensor,
    ):
        # Set the IMU to camera extrinsics.
        position_imu_in_camera = self.transformation_imu_to_camera[:3, 3].unsqueeze(0)
        rotation_imu_to_camera = lt.SO3(
            pp.from_matrix(self.transformation_imu_to_camera[:3, :3], ltype=pp.SO3_type).tensor().unsqueeze(0))

        if not self.imu_frame:
            # Transform the current control points from relative scale camera->world to absolute scale IMU->world.
            rotations_camera_to_world = lt.SO3(self.trajectory.rotation_spline.control_points.data)
            positions_imu_in_world = rotations_camera_to_world * position_imu_in_camera + \
                self.trajectory.position_spline.control_points.data * scale_factor
            rotations_imu_to_world = rotations_camera_to_world * rotation_imu_to_camera
        else:
            # Set the camera to IMU extrinsics.
            rotation_camera_to_imu = rotation_imu_to_camera.inv()
            position_camera_in_imu = -(rotation_camera_to_imu * position_imu_in_camera)

            # Transform the current position control points from absolute scale IMU->world to relative scale
            # camera->world using the current scale factor estimate.
            rotations_imu_to_world = lt.SO3(self.trajectory.rotation_spline.control_points.data)
            positions_camera_in_world = (rotations_imu_to_world * position_camera_in_imu +
                self.trajectory.position_spline.control_points.data) / self.trajectory.scale_factor.item()

            # Transform the position control points from relative scale camera->world back to absolute scale IMU->world
            # using the new scale factor estimate.
            rotations_camera_to_world = rotations_imu_to_world * rotation_camera_to_imu
            positions_imu_in_world = rotations_camera_to_world * position_imu_in_camera + \
                positions_camera_in_world * scale_factor

        # Initialize the new or updated IMU trajectory and the optimizers.
        self.imu_frame = True
        self.initialize_trajectory(
            position_control_points=positions_imu_in_world,
            rotation_control_points=rotations_imu_to_world.data,
            bias_gyro=bias_gyro,
            bias_accel=bias_accel,
            scale_factor=torch.tensor(scale_factor, dtype=torch.double),
            gravity_dir_world=gravity_dir_world)
        self.initialize_optimizers()

    def __call__(
        self,
        times_eval: torch.Tensor,
        as_matrices: bool = True,
        invert: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, lt.SO3]]:
        if torch.max(times_eval) > self.latest_time_eval:
            self.latest_time_eval = torch.max(times_eval).item()

        return self.trajectory(times_eval, as_matrices, invert)

    def zero_grad(
        self,
    ) -> None:
        self.position_optimizer.zero_grad(set_to_none=True)
        self.rotation_optimizer.zero_grad(set_to_none=True)
        if self.dataset.imu_available:
            self.bias_optimizer.zero_grad(set_to_none=True)
        if self.imu_frame:
            self.gravity_dir_optimizer.zero_grad(set_to_none=True)
            self.scale_optimizer.zero_grad(set_to_none=True)

    @NestedTimer.timed_function('IMU Loss Calculation')
    def get_loss(
        self,
        linear_accels_meas_imu: torch.Tensor,
        angular_vels_meas_imu: torch.Tensor,
        timestamps_imu: torch.Tensor,
        latest_random_timestamp: Optional[torch.Tensor] = None,
        weight_random: Optional[float] = None,
        weight_window: Optional[float] = None,
        use_reloc_weights = False,
    ) -> torch.Tensor:
        if not self.dataset.imu_available or \
            self.config.linear_accel_weight == 0.0 and self.config.angular_vel_weight == 0.0:
            return 0.0

        loss_linear_accels, loss_angular_vels, loss_bias_accel, loss_bias_gyro = self.trajectory.get_loss(
            linear_accels_meas_imu,
            angular_vels_meas_imu,
            timestamps_imu,
            latest_random_timestamp,
            weight_random,
            weight_window)
        if use_reloc_weights:
            loss = \
                loss_linear_accels * self.config.linear_accel_weight_reloc + \
                loss_angular_vels * self.config.angular_vel_weight_reloc
        else:
            loss = \
                loss_linear_accels * self.config.linear_accel_weight + \
                loss_angular_vels * self.config.angular_vel_weight
        loss += loss_bias_accel * self.config.bias_accel_weight + loss_bias_gyro * self.config.bias_gyro_weight

        return loss

    def step(
        self,
    ) -> None:
        self.position_optimizer.step()
        self.rotation_optimizer.step()
        if self.dataset.imu_available:
            self.bias_optimizer.step()
        if self.imu_frame:
            self.gravity_dir_optimizer.step()
            self.scale_optimizer.step()

    def freeze(
        self,
        poses=False,
        biases=False,
        gravity_dir=False,
        scale_factor=False,
    ) -> None:
        if poses:
            self.trajectory.position_spline.control_points.requires_grad = False
            self.trajectory.rotation_spline.control_points.requires_grad = False
        if biases and self.dataset.imu_available:
            self.trajectory.bias_gyro.requires_grad = False
            if self.imu_frame:
                self.trajectory.bias_accel.requires_grad = False
        if gravity_dir and self.imu_frame:
            self.trajectory.gravity_dir_world.requires_grad = False
        if scale_factor and self.imu_frame:
            self.trajectory.scale_factor.requires_grad = False

    def unfreeze(
        self,
        poses=False,
        biases=False,
        gravity_dir=False,
        scale_factor=False,
    ) -> None:
        if poses:
            self.trajectory.position_spline.control_points.requires_grad = True
            self.trajectory.rotation_spline.control_points.requires_grad = True
        if biases and self.dataset.imu_available:
            self.trajectory.bias_gyro.requires_grad = True
            if self.imu_frame:
                self.trajectory.bias_accel.requires_grad = True
        if gravity_dir and self.imu_frame:
            self.trajectory.gravity_dir_world.requires_grad = True
        if scale_factor and self.imu_frame:
            self.trajectory.scale_factor.requires_grad = True

    def get_copy(
        self,
        save_optimizer_states: bool = True,
        save_freeze_states: bool = True,
    ) -> Dict[str, Any]:
        manager_state = {
            'pose_spline_order': self.config.pose_spline_order,
            'pose_knot_interval': self.config.pose_knot_interval,
            'time_start': self.time_start,
            'latest_time_eval': self.latest_time_eval,
            'imu_frame': self.imu_frame}
        save_data = {
            'manager_state': manager_state,
            'parameter_states': copy.deepcopy(self.trajectory.state_dict())}

        if save_optimizer_states:
            save_data['optimizer_states'] = {}
            save_data['optimizer_states']['positions'] = copy.deepcopy(self.position_optimizer.state_dict())
            save_data['optimizer_states']['rotations'] = copy.deepcopy(self.rotation_optimizer.state_dict())
            if self.dataset.imu_available:
                save_data['optimizer_states']['biases'] = copy.deepcopy(self.bias_optimizer.state_dict())
            if self.imu_frame:
                save_data['optimizer_states']['gravity_dir'] = copy.deepcopy(self.gravity_dir_optimizer.state_dict())
                save_data['optimizer_states']['scale_factor'] = copy.deepcopy(self.scale_optimizer.state_dict())

        if save_freeze_states:
            save_data['freeze_states'] = {}
            save_data['freeze_states']['positions'] = self.trajectory.position_spline.control_points.requires_grad
            save_data['freeze_states']['rotations'] = self.trajectory.rotation_spline.control_points.requires_grad
            if self.dataset.imu_available:
                save_data['freeze_states']['biase_gyro'] = self.trajectory.bias_gyro.requires_grad
            if self.imu_frame:
                save_data['freeze_states']['bias_accel'] = self.trajectory.bias_accel.requires_grad
                save_data['freeze_states']['gravity_dir'] = self.trajectory.gravity_dir_world.requires_grad
                save_data['freeze_states']['scale_factor'] = self.trajectory.scale_factor.requires_grad

        return save_data

    def save(
        self,
        base_result_dir: str,
        suffix: Optional[str] = None,
        save_optimizer_states: bool = True,
        save_freeze_states: bool = True,
    ) -> None:
        result_dir = f'{base_result_dir}/trajectory'
        if suffix is not None:
            filename = f'trajectory_{suffix}.pt'
        else:
            filename = 'trajectory.pt'
        os.makedirs(result_dir, exist_ok=True)
        filepath = os.path.join(result_dir, filename)

        torch.save(self.get_copy(save_optimizer_states, save_freeze_states), filepath)

    def load_copy(
        self,
        save_data: Dict[str, Any],
        load_optimizer_states: bool = True,
        load_freeze_states: bool = True,
    ) -> None:
        assert \
            save_data['manager_state']['pose_spline_order'] == self.config.pose_spline_order and \
            save_data['manager_state']['pose_knot_interval'] == self.config.pose_knot_interval and \
            'The current and saved spline order and knot interval must match.'

        # Handle an earlier start time, if needed.
        old_time_start = save_data['manager_state']['time_start']
        new_time_start = self.time_start
        if new_time_start < old_time_start:
            print('The current start time is earlier than the stored start time. Duplicating the first control point '
                  'to extend the spline.')
            num_knots_needed = math.ceil((old_time_start - new_time_start) / self.config.pose_knot_interval)
            self.time_start = old_time_start - num_knots_needed * self.config.pose_knot_interval

            def extend_fn(old_tensor):
                new_elements = old_tensor[0].repeat(num_knots_needed, 1)
                return torch.cat([new_elements, old_tensor])
        else:
            extend_fn = lambda x: x

        self.latest_time_eval = save_data['manager_state']['latest_time_eval']
        self.imu_frame = save_data['manager_state']['imu_frame']

        trajectory_kwargs = {
            'position_control_points': extend_fn(save_data['parameter_states']['position_spline.control_points']),
            'rotation_control_points': extend_fn(save_data['parameter_states']['rotation_spline.control_points'])}
        if self.dataset.imu_available:
            trajectory_kwargs['bias_gyro'] = save_data['parameter_states']['bias_gyro']
            if self.imu_frame:
                trajectory_kwargs['bias_accel'] = save_data['parameter_states']['bias_accel']
                trajectory_kwargs['scale_factor'] = save_data['parameter_states']['scale_factor']
                trajectory_kwargs['gravity_dir_world'] = save_data['parameter_states']['gravity_dir_world']
        self.initialize_trajectory(**trajectory_kwargs)
        self.initialize_optimizers()

        if load_optimizer_states:
            if 'optimizer_states' in save_data:
                def extend_opt_state_fn(opt_state_dict):
                    param_state = opt_state_dict['state'][0]
                    for key in param_state.keys():
                        if isinstance(param_state[key], torch.Tensor) and param_state[key].dim() != 0:
                            param_state[key] = extend_fn(param_state[key])
                    return opt_state_dict

                self.position_optimizer.load_state_dict(extend_opt_state_fn(save_data['optimizer_states']['positions']))
                self.rotation_optimizer.load_state_dict(extend_opt_state_fn(save_data['optimizer_states']['rotations']))
                if self.dataset.imu_available:
                    self.bias_optimizer.load_state_dict(save_data['optimizer_states']['biases'])
                if self.imu_frame:
                    self.gravity_dir_optimizer.load_state_dict(save_data['optimizer_states']['gravity_dir'])
                    self.scale_optimizer.load_state_dict(save_data['optimizer_states']['scale_factor'])
            else:
                print('Warning: no trajectory optimizer states were saved. The trajectory optimizers have been '
                    'initialized with default states')

        if load_freeze_states:
            if 'freeze_states' in save_data:
                self.trajectory.position_spline.control_points.requires_grad = save_data['freeze_states']['positions']
                self.trajectory.rotation_spline.control_points.requires_grad = save_data['freeze_states']['rotations']
                if self.dataset.imu_available:
                    self.trajectory.bias_gyro.requires_grad = save_data['freeze_states']['biase_gyro']
                if self.imu_frame:
                    self.trajectory.bias_accel.requires_grad = save_data['freeze_states']['bias_accel']
                    self.trajectory.gravity_dir_world.requires_grad = save_data['freeze_states']['gravity_dir']
                    self.trajectory.scale_factor.requires_grad = save_data['freeze_states']['scale_factor']
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
        result_dir = f'{base_result_dir}/trajectory'
        if suffix is not None:
            filename = f'trajectory_{suffix}.pt'
        else:
            filename = 'trajectory.pt'
        filepath = os.path.join(result_dir, filename)
        save_data = torch.load(filepath, weights_only=False, map_location=self.config.device)

        self.load_copy(save_data, load_optimizer_states, load_freeze_states)
