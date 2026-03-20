from typing import Dict, Union, Type, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
import cv2
import torch

@dataclass
class BaseDatasetConfig:
    """Base dataset config. Derived config classes must set '_data_class' to the corresponding derived dataset class."""

    _dataset_class: Type = field(default_factory=lambda: BaseDataset)
    # The time (in seconds) at which to begin processing images. If -1.0, processing will begin with the first image.
    time_begin: float = -1.0
    # The time (in seconds) at which to stop processing images. If -1.0, processing will continue until the last image.
    time_end: float = -1.0
    # The factor used in downsampling images. Block averaging is used for downsampling, so the image dimensions must be
    # divisible by this downsampling factor.
    downsample_factor: int = 1

class BaseDataset(ABC):
    """Base class for microbolometer thermal datasets. Derived classes must implement each of the abstract properties
    and functions."""

    _new_camera_matrix: NDArray = None
    _mapx: NDArray = None
    _mapy: NDArray = None

    # Abstract properties and functions.

    @property
    @abstractmethod
    def image_width_full_res(self) -> int:
        """Returns the image width at full resolution."""
        pass

    @property
    @abstractmethod
    def image_height_full_res(self) -> int:
        """Returns the image height at full resolution."""
        pass

    @property
    @abstractmethod
    def average_frame_period(self) -> float:
        """Returns the average frame period (in seconds)."""
        pass

    @property
    @abstractmethod
    def transformation_imu_to_camera(self) -> Optional[NDArray]:
        """Returns the transformation matrix that takes a point from the IMU frame to the camera frame, following the
        camera frame convention (x-right, y-down, z-forward).

        If the dataset does not include an IMU, this should return None.
        """
        pass

    @property
    @abstractmethod
    def readout_row_duration_full_res(self) -> float:
        """Returns the time it takes the camera to readout one row at full resolution (in seconds).

        If the camera is global shutter, this should return 0.
        """
        pass

    @property
    @abstractmethod
    def thermal_time_constant(self) -> float:
        """Returns the thermal time constant (in seconds)."""
        pass

    @property
    @abstractmethod
    def original_camera_matrix_full_res(self) -> NDArray:
        """Returns the full resolution camera matrix."""
        pass

    @property
    @abstractmethod
    def distortion_model(self) -> str:
        """Returns the distortion model (either 'radtan' or 'equidistant', as defined by Kalibr)."""
        pass

    @property
    @abstractmethod
    def distortion_coefficients(self) -> NDArray:
        """Returns the distortion coeffcients."""
        pass

    @property
    @abstractmethod
    def downsample_factor(self) -> int:
        """Returns the image downsample factor."""
        pass

    @property
    @abstractmethod
    def image_timestamps(self) -> NDArray:
        """Returns the image timestamps (in seconds) as a 1D numpy array with dtype np.float64.

        Each image timestamp corresponds to the time at which the first pixel is read out.
        """
        pass

    @property
    @abstractmethod
    def accel_noise_density(self) -> Optional[float]:
        """Returns the accelerometer noise density (in radians / second^0.5).

        Note that calibrated values tend to be too small and should typically be multiplied by some factor (e.g. 10).

        If the dataset does not include an IMU, this should return None.
        """
        pass

    @property
    @abstractmethod
    def gyro_noise_density(self) -> Optional[float]:
        """Returns the gyroscope noise density (in meters / second^1.5).

        Note that calibrated values tend to be too small and should typically be multiplied by some factor (e.g. 10).

        If the dataset does not include an IMU, this should return None.
        """
        pass

    @property
    @abstractmethod
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
        pass

    @property
    @abstractmethod
    def ground_truth_poses(self) -> Optional[NDArray]:
        """Returns the ground truth poses.

        If the dataset does not include ground truth poses, this should return None.

        Otherwise, the ground truth poses should be a numpy array with dtype np.float64 and shape [num_images, 4, 4]
        containing transformation matrices that take a point from the camera frame to the world frame, following the
        camera frame convention (x-right, y-down, z-forward).
        """
        pass

    @abstractmethod
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
        pass

    # Concrete properties and functions.

    @property
    def image_width(self) -> int:
        """Returns the image width after downsampling."""
        return self.image_width_full_res // self.downsample_factor

    @property
    def image_height(self) -> int:
        """Returns the image height after downsampling."""
        return self.image_height_full_res // self.downsample_factor

    @property
    def original_camera_matrix(self) -> NDArray:
        """Returns the camera matrix after downsampling."""
        original_camera_matrix = self.original_camera_matrix_full_res.copy()
        original_camera_matrix[:2, :3] = original_camera_matrix[:2, :3] / self.downsample_factor
        return original_camera_matrix

    @property
    def readout_row_duration(self) -> float:
        """Returns the time it takes the camera to readout one row after downsampling (in seconds)."""
        return self.readout_row_duration_full_res * self.downsample_factor

    @property
    def new_camera_matrix(self) -> NDArray:
        """Returns the new camera matrix."""
        if self._new_camera_matrix is not None:
            return self._new_camera_matrix

        if self.distortion_model == 'radtan':
            # Using alpha = 0 to ensure no invalid pixels.
            self._new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.original_camera_matrix,
                self.distortion_coefficients, (self.image_width, self.image_height), alpha=0)
        elif self.distortion_model == 'equidistant':
            # Using balance = 0 to ensure no invalid pixels.
            self._new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.original_camera_matrix, self.distortion_coefficients, (self.image_width, self.image_height),
                np.eye(3), balance=0)
        else:
            raise ValueError(f'Unsupported distortion model: {self.distortion_model}.')

        return self._new_camera_matrix

    def compute_undistortion_maps(self) -> None:
        """Computes the undistortion maps."""
        # Using CV_32FC1 for slower but more accurate remapping, see OpenCV's documentation for convertMaps().
        if self.distortion_model == 'radtan':
            self._mapx, self._mapy = cv2.initUndistortRectifyMap(self.original_camera_matrix,
                self.distortion_coefficients, np.eye(3), self.new_camera_matrix, (self.image_width, self.image_height),
                cv2.CV_32FC1)
        elif self.distortion_model == 'equidistant':
            self._mapx, self._mapy = cv2.fisheye.initUndistortRectifyMap(self.original_camera_matrix,
                self.distortion_coefficients, np.eye(3), self.new_camera_matrix, (self.image_width, self.image_height),
                cv2.CV_32FC1)
        else:
            raise ValueError(f'Unsupported distortion model: {self.distortion_model}.')

    @property
    def mapx(self) -> NDArray:
        """Returns the first undistortion map."""
        if self._mapx is None:
            self.compute_undistortion_maps()
        return self._mapx

    @property
    def mapy(self) -> NDArray:
        """Returns the second undistortion map."""
        if self._mapy is None:
            self.compute_undistortion_maps()
        return self._mapy

    @property
    def imu_available(self) -> bool:
        """Returns a bool indicating whether IMU data is available."""
        return self.transformation_imu_to_camera is not None

    def __len__(self) -> int:
        """Returns the number of images."""
        return len(self.image_timestamps)

    def downsample_block_averaging(
        self,
        image: NDArray,
    ) -> NDArray:
        """Downsamples an image using block averaging.

        Args:
            image: The input image with shape [height, width]. The image dimensions must be divisible by the downsample
                factor.

        Returns:
            The downsampled image with shape [height/downsample_factor, width/downsample_factor]. The output image has
            the same dtype as the input image.
        """
        height, width = image.shape

        assert height % self.downsample_factor == 0 and width % self.downsample_factor == 0, 'The image dimensions ' \
            f'({height}x{width}) are not divisible by the downsample factor ({self.downsample_factor}).'

        reshaped_image = image.reshape(
            height // self.downsample_factor,
            self.downsample_factor,
            width // self.downsample_factor,
            self.downsample_factor)
        downsampled_image = reshaped_image.astype(np.float64).mean(axis=(1, 3)).astype(image.dtype)

        return downsampled_image

    def get_imu_data_between_images(
        self,
        index_start,
        index_stop,
    ) -> Dict[str, NDArray]:
        """Returns the IMU measurements made between two images.

        Note that the returned IMU measurements are specifically those that were made after the last pixel in the first
        image and before (or at the same time as) the last pixel in the second image.

        Args:
            index_start: The first image index.
            index_stop: The second image index.

        Returns:
            A dictionary containing the IMU data. The dictionary has the same format as `self.imu_data`.
        """
        total_readout_time = self.image_height * self.readout_row_duration
        if index_start >= 0:
            first_imu_timestamp = self.image_timestamps[index_start] + total_readout_time
        else:
            first_imu_timestamp = self.image_timestamps[0]
        last_imu_timestamp = self.image_timestamps[index_stop] + total_readout_time

        imu_selection_mask = \
            (self.imu_data['timestamps'] > first_imu_timestamp) & \
            (self.imu_data['timestamps'] <= last_imu_timestamp)

        selected_imu_data = {
            'linear_accels': self.imu_data['linear_accels'][imu_selection_mask],
            'angular_vels': self.imu_data['angular_vels'][imu_selection_mask],
            'timestamps': self.imu_data['timestamps'][imu_selection_mask]}

        return selected_imu_data

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Union[int, torch.Tensor]]:
        """Returns data corresponding to the indexed image.

        The data is returned as a dictionary that contains the following:
        - 'index': The image index (same as the input index) as an int.
        - 'image': The undistorted image as a torch tensor with dtype torch.float and shape [height, width].
        - 'image_timestamp': The image timestamp (in seconds) corresponding to the time at which the first pixel is
            read out as a scalar torch tensor with dtype torch.double.

        If the dataset includes an IMU, the dictionary will additionally include the following:
        -'linear_accels': The accelerometer measurements (proper acceleration measurements of the IMU expressed in the
            IMU frame, in units of meters/second^2), made between the previous and current image, as a torch tensor with
            dtype torch.double and shape [num_measurements, 3].
        - 'angular_vels': The gyroscope measurements (angular velocity measurements of the IMU expressed in the IMU
            frame, in units of radians/second), made between the previous and current image, as a torch tensor with
            dtype torch.double and shape [num_measurements, 3].
        - 'imu_timestamps': The IMU measurement timestamps (in seconds) as a 1D torch tensor with dtype torch.double.

        Args:
            index: The image index.

        Returns:
            The data dictionary as described above.
        """
        image_data = self._get_image_data(index)
        image = self.downsample_block_averaging(image_data['image'])
        image = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)

        output_data = {
            'index': image_data['index'],
            'image': torch.from_numpy(image).float().clamp(0.0, 1.0),
            'image_timestamp': torch.tensor(image_data['timestamp'], dtype=torch.double)}

        if self.imu_available:
            imu_data = self.get_imu_data_between_images(index - 1, index)

            output_data.update({
                'linear_accels': torch.from_numpy(imu_data['linear_accels']).double(),
                'angular_vels': torch.from_numpy(imu_data['angular_vels']).double(),
                'imu_timestamps': torch.from_numpy(imu_data['timestamps']).double()})

        return output_data
