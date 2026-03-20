from typing import Dict, Union, Type, Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import yaml
import h5py
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from trgs_slam.datasets.base_dataset import BaseDatasetConfig, BaseDataset

def read_yaml(path_yaml_file):
    with open(path_yaml_file, 'r') as file:
        info = yaml.safe_load(file)
    return info

def get_camera_index(calibration_results, camera_name):
    for key in calibration_results:
        if camera_name in calibration_results[key]['rostopic'].replace('/', '_'):
            return int(key.replace('cam', ''))
    print(f'The camera {camera_name} was not found in the calibration file')
    exit()

def invert_transformation_matrix(transformation_matrix):
    transformation_matrix_inverse = np.eye(4)
    transformation_matrix_inverse[0:3, 0:3] = transformation_matrix[0:3, 0:3].T
    transformation_matrix_inverse[0:3, 3] = -transformation_matrix_inverse[0:3, 0:3] @ transformation_matrix[0:3, 3]

    return transformation_matrix_inverse

def get_transformation_from_calib(calibration_results, source_camera_name, target_camera_name):
    # Identify source and target camera indices in the calibration results
    index_source_camera = get_camera_index(calibration_results, source_camera_name)
    index_target_camera = get_camera_index(calibration_results, target_camera_name)

    # Get the transformation matrix from the source camera to the target camera
    transformation_source_to_target = np.eye(4)
    for index_camera in range(np.min([index_source_camera, index_target_camera]) + 1,
        np.max([index_source_camera, index_target_camera]) + 1):
        transformation_prev_to_current = np.asarray(calibration_results['cam' + str(index_camera)]['T_cn_cnm1'])
        transformation_source_to_target = transformation_prev_to_current @ transformation_source_to_target
    if index_target_camera < index_source_camera:
        transformation_source_to_target = invert_transformation_matrix(transformation_source_to_target)

    return transformation_source_to_target

@dataclass
class TRNeRFDatasetConfig(BaseDatasetConfig):
    _dataset_class: Type = field(default_factory=lambda: TRNeRFDataset)
    # Path to H5 file with the thermal images to process.
    path_images: str = ''
    # Path to Kalibr camchain YAML file.
    path_calibration_results: str = ''
    # Path to H5 file with IMU data to process. If this and both IMU calibration files are provided, this will be used
    # in pose optimization.
    path_imu: Optional[str] = None
    # Path to IMU-camera calibration results giving the transformation between the mono left camera and the IMU.
    path_imu_cam_calibration: Optional[str] = None
    # Path to IMU noise calibration results.
    path_imu_noise_calibration: Optional[str] = None
    # Path to ground truth poses.
    path_ground_truth_poses: Optional[str] = None
    # Factor to multiply IMU noise parameters by.
    imu_noise_factor: float = 10.0
    # The thermal time constant (in seconds).
    thermal_time_constant: float = 8e-3
    # The delay between image capture being triggered and the beginning of readout (in seconds).
    readout_delay: float = 0.5e-3
    # The time it takes the camera to readout one row (in seconds).
    readout_row_duration: float = 27.8e-6
    # Minimum threshold used for remapping thermal image values.
    threshold_minimum: float = 0.0
    # Maximum threshold used for remapping thermal image values.
    threshold_maximum: float = 2**16 - 1

class TRNeRFDataset(BaseDataset):
    """TRNeRF dataset."""

    def __init__(
        self,
        config: TRNeRFDatasetConfig,
    ) -> None:
        self.config = config
        assert self.config.path_images != '', 'A path to image data is required.'
        assert self.config.path_calibration_results != '', 'A path to calibration results is required.'

        # Read in calibration results.
        calibration_results = read_yaml(self.config.path_calibration_results)

        # Open H5 image file.
        h5_image_file = h5py.File(self.config.path_images, 'r')
        group_image = h5_image_file['image_raw']
        self.image_data_dict = {
            'name': Path(self.config.path_images).parts[-1][:-3],
            'images': group_image['images'],
            'timestamps': group_image['timestamps'][:]}

        # Determine the timestamps of the images to process and the average frame period.
        all_image_timestamps = self.image_data_dict['timestamps'] * 1e-9
        time_begin = self.config.time_begin if self.config.time_begin != -1.0 else all_image_timestamps[0]
        time_end = self.config.time_end if self.config.time_end != -1.0 else all_image_timestamps[-1]
        image_selection_mask = (all_image_timestamps >= time_begin) & (all_image_timestamps <= time_end)
        self.selected_image_indices = np.where(image_selection_mask)[0]
        self._image_timestamps = \
            (all_image_timestamps[image_selection_mask] + self.config.readout_delay).astype(np.float64)
        self._average_frame_period = np.mean(all_image_timestamps[1:] - all_image_timestamps[:-1])

        # Unpack the intrinsics.
        index_camera = get_camera_index(calibration_results, self.image_data_dict['name'])
        calibration_result = calibration_results['cam' + str(index_camera)]
        intrinsics = calibration_result['intrinsics']
        self._original_camera_matrix_full_res = np.eye(3)
        self._original_camera_matrix_full_res[0, 0] = intrinsics[0]
        self._original_camera_matrix_full_res[1, 1] = intrinsics[1]
        self._original_camera_matrix_full_res[0, 2] = intrinsics[2]
        self._original_camera_matrix_full_res[1, 2] = intrinsics[3]
        self._distortion_coefficients = np.array(calibration_result['distortion_coeffs'])
        self._distortion_model = calibration_result['distortion_model']
        resolution = calibration_result['resolution']
        self._image_width_full_res = resolution[0]
        self._image_height_full_res = resolution[1]

        # Set the thermal camera parameters.
        self._readout_row_duration_full_res = self.config.readout_row_duration
        self._thermal_time_constant = self.config.thermal_time_constant

        # Set the downsample factor.
        self._downsample_factor = self.config.downsample_factor

        # Set the image thresholds.
        self.threshold_minimum = self.config.threshold_minimum
        self.threshold_maximum = self.config.threshold_maximum

        load_imu_data = self.config.path_imu is not None and \
            self.config.path_imu_cam_calibration is not None and \
            self.config.path_imu_noise_calibration is not None
        if load_imu_data:
            # Compute the transformation from the IMU to the camera.
            transformation_mono_left_to_camera = get_transformation_from_calib(calibration_results, 'mono_left',
                self.image_data_dict['name'])
            calibration_info_imu_cam = read_yaml(self.config.path_imu_cam_calibration)
            transformation_imu_to_mono_left = np.array(calibration_info_imu_cam['transformation_imu_to_mono_left'])
            self._transformation_imu_to_camera = transformation_mono_left_to_camera @ transformation_imu_to_mono_left

            # Read in IMU data.
            h5_imu_file = h5py.File(self.config.path_imu, 'r')
            group_imu = h5_imu_file['imu']
            self._imu_data = {
                'linear_accels': group_imu['linear_accelerations'][:].astype(np.float64),
                'angular_vels': group_imu['angular_velocities'][:].astype(np.float64),
                'timestamps': group_imu['timestamps'][:].astype(np.float64) * 1e-9}

            # Read in the IMU noise parameters.
            calibration_info_imu_noise = read_yaml(self.config.path_imu_noise_calibration)
            self._accel_noise_density = \
                calibration_info_imu_noise['accelerometer_noise_density'] * self.config.imu_noise_factor
            self._gyro_noise_density = \
                calibration_info_imu_noise['gyroscope_noise_density'] * self.config.imu_noise_factor
        else:
            self._transformation_imu_to_camera = None
            self._imu_data = None
            self._accel_noise_density = None
            self._gyro_noise_density = None

        if self.config.path_ground_truth_poses is not None:
            # Read in the ground truth poses from the world to the left monochrome camera.
            h5_pose_file = h5py.File(self.config.path_ground_truth_poses, 'r')
            group_poses = h5_pose_file['poses_mono_left']

            positions_world_in_mono_left = group_poses['positions'][:]
            rotations_world_to_mono_left = Rotation.from_quat(group_poses['quaternions'][:])
            pose_timestamps = group_poses['timestamps'][:] * 1e-9

            # Compute the ground truth poses from the camera to be processed to the world.
            positions_mono_left_in_world = -rotations_world_to_mono_left.inv().apply(positions_world_in_mono_left)
            rotations_mono_left_to_world = rotations_world_to_mono_left.inv()
            transformations_mono_left_to_world = np.tile(np.eye(4), (len(pose_timestamps), 1, 1))
            transformations_mono_left_to_world[:, :3, :3] = rotations_mono_left_to_world.as_matrix()
            transformations_mono_left_to_world[:, :3, 3] = positions_mono_left_in_world

            transformation_camera_to_mono_left = get_transformation_from_calib(calibration_results,
                self.image_data_dict['name'], 'mono_left')
            transformations_camera_to_world = transformations_mono_left_to_world @ transformation_camera_to_mono_left

            # Interpolate the poses at the image timestamps.
            position_interpolator = interp1d(pose_timestamps, transformations_camera_to_world[:, :3, 3], axis=0)
            interpolated_positions = position_interpolator(self.image_timestamps)
            rotation_interpolator = \
                Slerp(pose_timestamps, Rotation.from_matrix(transformations_camera_to_world[:, :3, :3]))
            interpolated_rotations = rotation_interpolator(self.image_timestamps)

            self._ground_truth_poses = np.tile(np.eye(4), (len(self.image_timestamps), 1, 1))
            self._ground_truth_poses[:, :3, :3] = interpolated_rotations.as_matrix()
            self._ground_truth_poses[:, :3, 3] = interpolated_positions
        else:
            self._ground_truth_poses = None

    @property
    def image_width_full_res(self) -> int:
        """Returns the image width at full resolution."""
        return self._image_width_full_res

    @property
    def image_height_full_res(self) -> int:
        """Returns the image height at full resolution."""
        return self._image_height_full_res

    @property
    def average_frame_period(self) -> float:
        """Returns the average frame period (in seconds)."""
        return self._average_frame_period

    @property
    def transformation_imu_to_camera(self) -> Optional[NDArray]:
        """Returns the transformation matrix that takes a point from the IMU frame to the camera frame, following the
        camera frame convention (x-right, y-down, z-forward).

        If the dataset does not include an IMU, this should return None.
        """
        return self._transformation_imu_to_camera

    @property
    def readout_row_duration_full_res(self) -> float:
        """Returns the time it takes the camera to readout one row at full resolution (in seconds).

        If the camera is global shutter, this should return 0.
        """
        return self._readout_row_duration_full_res

    @property
    def thermal_time_constant(self) -> float:
        """Returns the thermal time constant (in seconds)."""
        return self._thermal_time_constant

    @property
    def original_camera_matrix_full_res(self) -> NDArray:
        """Returns the full resolution camera matrix."""
        return self._original_camera_matrix_full_res

    @property
    def distortion_model(self) -> str:
        """Returns the distortion model (either 'radtan' or 'equidistant', as defined by Kalibr)."""
        return self._distortion_model

    @property
    def distortion_coefficients(self) -> NDArray:
        """Returns the distortion coeffcients."""
        return self._distortion_coefficients

    @property
    def downsample_factor(self) -> int:
        """Returns the image downsample factor."""
        return self._downsample_factor

    @property
    def image_timestamps(self) -> NDArray:
        """Returns the image timestamps (in seconds) as a 1D numpy array with dtype np.float64.

        Each image timestamp corresponds to the time at which the first pixel is read out.
        """
        return self._image_timestamps

    @property
    def accel_noise_density(self) -> Optional[float]:
        """Returns the accelerometer noise density (in radians / second^0.5).

        Note that calibrated values tend to be too small and should typically be multiplied by some factor (e.g. 10).

        If the dataset does not include an IMU, this should return None.
        """
        return self._accel_noise_density

    @property
    def gyro_noise_density(self) -> Optional[float]:
        """Returns the gyroscope noise density (in meters / second^1.5).

        Note that calibrated values tend to be too small and should typically be multiplied by some factor (e.g. 10).

        If the dataset does not include an IMU, this should return None.
        """
        return self._gyro_noise_density

    @property
    def imu_data(self) -> Optional[Dict[str, NDArray]]:
        """Returns the IMU data.

        If the dataset does not include an IMU, this should return None.

        Otherwise, the IMU data is a dictionary that includes the following:
        - 'linear_accels': The accelerometer measurements (proper acceleration measurements of the IMU expressed in the
            IMU frame, in units of meters/second^2) as a numpy array with dtype np.float64 and shape
            [num_measurements, 3].
        - 'angular_vels': The gyroscope measurements (angular velocity measurements of the IMU expressed in the IMU
            frame, in units of radians/second) as a numpy array with dtype np.float64 and shape [num_measurements, 3].
        - 'timestamps': The IMU measurement timestamps (in seconds) as a 1D numpy array with dtype np.float64.
        """
        return self._imu_data

    @property
    def ground_truth_poses(self) -> Optional[NDArray]:
        """Returns the ground truth poses.

        If the dataset does not include ground truth poses, this should return None.

        Otherwise, the ground truth poses should be a numpy array with dtype np.float64 and shape [num_images, 4, 4]
        containing transformation matrices that take a point from the camera frame to the world frame, following the
        camera frame convention (x-right, y-down, z-forward).
        """
        return self._ground_truth_poses

    def _get_image_data(
        self,
        index: int,
    ) -> Dict[str, Union[int, NDArray]]:
        """Returns data corresponding to the indexed image.

        The data must be returned as a dictionary that contains the following:
        - 'index': The image index (same as the input index).
        - 'image': The raw (distorted) image as a numpy array, remapped to the range [0, 1] with dtype np.float32 and
            shape [height, width].
        - 'timestamp': The image timestamp (in seconds) corresponding to the time at which the first pixel is
            read out.

        Args:
            index: The image index.

        Returns:
            The data dictionary as described above.
        """
        timestamp = self.image_timestamps[index]

        image = self.image_data_dict['images'][self.selected_image_indices[index]]
        image[image < self.threshold_minimum] = self.threshold_minimum
        image[image > self.threshold_maximum] = self.threshold_maximum
        image = (image.astype(np.float32) - self.threshold_minimum) / (self.threshold_maximum - self.threshold_minimum)

        image_data = {
            'index': index,
            'image': image,
            'timestamp': timestamp}

        return image_data
