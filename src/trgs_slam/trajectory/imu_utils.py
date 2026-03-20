from typing import Dict, Union, Tuple

import math
import numpy as np
import torch

import pypose as pp
import lietorch as lt

GRAVITY_MAGNITUDE = 9.80665

def preintegrate_imu(
    linear_accels_meas_imu: torch.Tensor,
    angular_vels_meas_imu: torch.Tensor,
    timestamps_imu: torch.Tensor,
    t_j: float,
    accelerometer_noise_density: float,
    gyroscope_noise_density: float,
    bias_accel: torch.Tensor = torch.zeros(3),
    bias_gyro: torch.Tensor = torch.zeros(3),
) -> Dict[str, Union[pp.LieTensor, torch.Tensor]]:
    """Perform IMU preintegration.

    The preintegration is performed from t_i (timestamps_imu[0]) to t_j.

    The implementation follows the paper:
    C. Forster, et al., "On-Manifold Preintegration for Real-Time Visual–Inertial Odometry.", TRO 2017
    Many of the variable names have been chosen to match and references to equation numbers in the comments are
    referring to this paper.

    Note that the above paper focuses on computing the returned quantities iteratively, while here we operate on a batch
    of IMU data. The batched calculations for the preintegrated measurements and the covariance are inspired by the
    methods used in the IMUPreintegrator available in Pypose:
    C. Wang, et al., "PyPose: A library for robot learning with physics-based optimization.", CVPR 2023

    Args:
        linear_accels_meas_imu: Accelerometer measurements (in meters/second^2) (shape [num_imu_measurements, 3]).
        angular_vels_meas_imu: Gyroscope measurements (in radians/second) (shape [num_imu_measurements, 3]).
        timestamps_imu: IMU measurement timestamps (in seconds) (shape [num_imu_measurements]).
        t_j: The timestamp (in seconds) defining the end of the preintegration interval.
        accelerometer_noise_density: Continuous time accelerometer noise density (in meters/(second^2 sqrt(Hz)) or,
            equivalently, meters/second^1.5).
        gyroscope_noise_density: Continuous time gyroscope noise density (in radians/(second sqrt(Hz)) or,
            equivalently, radians/sqrt(second)).
        bias_accel: Accelerometer bias (in meters/second^2) (shape [3]).
        bias_gyro: Gyroscope bias (in radians/second) (shape [3]).

    Returns:
        A dictionary with the following items (all Tensors, except Delta_R_ij, which is a SO3Type LieTensor):
            'Delta_R_ij': The preintegrated rotation measurement.
            'Delta_v_ij': The preintegrated velocity measurement.
            'Delta_p_ij': The preintegrated position measurement.
            'J_R_g_ij': The Jacobian of the preintegrated rotation measurement with respect to the gyroscope bias.
            'J_v_a_ij': The Jacobian of the preintegrated velocity measurement with respect to the accelerometer bias.
            'J_v_g_ij': The Jacobian of the preintegrated velocity measurement with respect to the gyroscope bias.
            'J_p_a_ij': The Jacobian of the preintegrated position measurement with respect to the accelerometer bias.
            'J_p_g_ij': The Jacobian of the preintegrated position measurement with respect to the gyroscope bias.
            'Sigma_ij': The preintegrated noise covariance.
    """
    device = linear_accels_meas_imu.device
    dtype = linear_accels_meas_imu.dtype
    bias_accel = bias_accel.to(device=device, dtype=dtype)
    bias_gyro = bias_gyro.to(device=device, dtype=dtype)

    # Compute the time deltas.
    # In the paper, the time delta is assumed to be constant. Here, we compute the delta between each IMU measurement
    # and the time delta between the last IMU measurement and t_j.
    Delta_ts = torch.cat([timestamps_imu[1:] - timestamps_imu[:-1], (t_j - timestamps_imu[-1]).unsqueeze(0)]
        ).to(dtype=dtype).unsqueeze(-1)

    # Remove the bias from the gyroscope measurments.
    unbiased_angular_vels = angular_vels_meas_imu - bias_gyro

    # Compute the rotation increments at each iteration.
    Delta_R_is = pp.so3(unbiased_angular_vels * Delta_ts).Exp()

    # Compute the preintegrated rotation measurements:
    # \Delta R_{ik} (equation 33, without noise removal) for k = i to k = j where R_{ii} is identity.
    Delta_R_ii = pp.identity_SO3(1, device=device, dtype=dtype)
    Delta_R_iks = \
        pp.cumprod(
            torch.cat([
                Delta_R_ii,
                Delta_R_is]),
        dim=0, left=False)

    # Remove the bias from the accelerometer measurments.
    unbiased_linear_accels = linear_accels_meas_imu - bias_accel

    # Compute the preintegrated velocity measurements:
    # \Delta v_{ik} (equation 33, without noise removal) for k = i to k = j where v_{ii} is zero.
    Delta_v_ii = torch.zeros(1, 3, device=device, dtype=dtype)
    Delta_v_iks = \
        torch.cumsum(
            torch.cat([
                Delta_v_ii,
                (Delta_R_iks[:-1] @ unbiased_linear_accels) * Delta_ts]),
        dim=0)

    # Compute the preintegrated position measurement:
    # \Delta p_{ij} (equation 33, without noise removal).
    Delta_p_ij = \
        torch.sum(
            Delta_v_iks[:-1] * Delta_ts + 0.5 * (Delta_R_iks[:-1] @ unbiased_linear_accels) * Delta_ts**2,
        dim=0)

    # Compute the Jacobians of the preintegrated rotation measurements with respect to the gyroscope bias:
    # \frac{\partial \Delta R_{ik}}{\partial b^g} for k = i to k = j where at k = i it is zero.
    # The first equation at the end of Appendix B is an expression for this.
    # The iterative form is found similarly to equation 59.
    @torch.jit.script
    def compute_J_R_g_iks(
        Delta_R_is_inv_mat: torch.Tensor,
        Delta_R_is_Jr: torch.Tensor,
        Delta_ts: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        J_R_g_iks = [torch.zeros(1, 3, 3, device=device, dtype=dtype)]
        for i in range(len(Delta_ts)):
            J_R_g_iks.append(Delta_R_is_inv_mat[i] @ J_R_g_iks[-1] - Delta_R_is_Jr[i] * Delta_ts[i])
        return torch.cat(J_R_g_iks)
    Delta_ts = Delta_ts.unsqueeze(-1)
    J_R_g_iks = compute_J_R_g_iks(Delta_R_is.Inv().matrix(), Delta_R_is.Jr(), Delta_ts, device, dtype)

    # Convert the unbiased linear acceleration from vectors to skew symmetric matrices.
    unbiased_linear_accels_skew = pp.vec2skew(unbiased_linear_accels)

    # Convert the preintegrated rotation measurements from quaternions to rotation matrices.
    Delta_R_iks_mat = Delta_R_iks.matrix()

    # Compute the Jacobians of the preintegrated velocity measurements with respect to the accelerometer bias:
    # \frac{\partial \Delta v_{ik}}{\partial b^a} for k = i to k = j where at k = i it is zero.
    # This follows the second equation at the end of Appendix B.
    J_v_a_ii = torch.zeros(1, 3, 3, device=device, dtype=dtype)
    J_v_a_iks = \
        torch.cumsum(
            torch.cat([
                J_v_a_ii,
                -Delta_R_iks_mat[:-1] * Delta_ts]),
        dim=0)

    # Compute the Jacobians of the preintegrated velocity measurements with respect to the gyroscope bias:
    # \frac{\partial \Delta v_{ik}}{\partial b^g} for k = i to k = j where at k = i it is zero.
    # This follows the third equation at the end of Appendix B.
    J_v_g_ii = torch.zeros(1, 3, 3, device=device, dtype=dtype)
    J_v_g_iks = \
        torch.cumsum(
            torch.cat([
                J_v_g_ii,
                -Delta_R_iks_mat[:-1] @ unbiased_linear_accels_skew @ J_R_g_iks[:-1] * Delta_ts]),
        dim=0)

    # Compute the Jacobian of the preintegrated position measurement with respect to the accelerometer bias:
    # \frac{\partial \Delta p_{ij}}{\partial b^a}.
    # This follows the fourth equation at the end of Appendix B.
    J_p_a_ij = \
        torch.sum(
            J_v_a_iks[:-1] * Delta_ts - 0.5 * Delta_R_iks_mat[:-1] * Delta_ts**2,
        dim=0)

    # Compute the Jacobian of the preintegrated position measurement with respect to the gyroscope bias:
    # \frac{\partial \Delta p_{ij}}{\partial b^g}.
    # This follows the fifth equation at the end of Appendix B.
    J_p_g_ij = \
        torch.sum(
            J_v_g_iks[:-1] * Delta_ts -
                0.5 * Delta_R_iks_mat[:-1] @ unbiased_linear_accels_skew @ J_R_g_iks[:-1] * Delta_ts**2,
        dim=0)

    # Compute the covariance matrices for the raw IMU measurement noise.
    # \Sigma_{\eta} (used in equation 63).
    # As mentioned at the end of section V, the covariance of the discrete time noise is equal to the covariance of the
    # continuous time noise, multiplied by one over the time delta.
    # In the paper, the time delta is assumed to be constant. Here, we compute the covariance matrices for each time
    # delta.
    Sigma_raw_ks = torch.eye(6, device=device, dtype=dtype).repeat(len(Delta_ts), 1, 1)
    Sigma_raw_ks[:, :3, :3] = torch.eye(3, dtype=dtype, device=device) * (1 / Delta_ts) * gyroscope_noise_density**2
    Sigma_raw_ks[:, 3:, 3:] = torch.eye(3, dtype=dtype, device=device) * (1 / Delta_ts) * accelerometer_noise_density**2

    # Compute the covariance.
    # \Sigma_{ij}.
    # Equation 63 gives an iterative expression for this.
    # Here, we unroll the iterative expression to compute it efficiently.
    A_ks = torch.eye(9, device=device, dtype=dtype).repeat(len(Delta_ts), 1, 1)
    A_ks[:-1, 0:3, 0:3] = Delta_R_is[1:].matrix().mT
    A_ks[:-1, 3:6, 0:3] = -Delta_ts[1:] * Delta_R_iks_mat[1:-1] @ unbiased_linear_accels_skew[1:]
    A_ks[:-1, 6:9, 0:3] = -0.5 * Delta_ts[1:]**2 * Delta_R_iks_mat[1:-1] @ unbiased_linear_accels_skew[1:]
    A_ks[:-1, 6:9, 3:6] = torch.eye(3, dtype=dtype, device=device).repeat(len(Delta_ts) - 1, 1, 1) * Delta_ts[1:]

    B_ks = torch.zeros(len(Delta_ts), 9, 6, device=device, dtype=dtype)
    B_ks[:, 0:3, 0:3] = Delta_R_is.Jr() * Delta_ts
    B_ks[:, 3:6, 3:6] = Delta_R_iks_mat[:-1] * Delta_ts
    B_ks[:, 6:9, 3:6] = 0.5 * Delta_R_iks_mat[:-1] * Delta_ts**2

    second_term_ks = B_ks @ Sigma_raw_ks @ B_ks.mT # The second term of eq. 63 at each iteration.
    A_left_cumulative = pp.cumprod(A_ks.flip([0]), dim=0, left=False).flip([0])
    A_right_cumulative = A_left_cumulative.mT
    Sigma_ij = torch.sum(A_left_cumulative @ second_term_ks @ A_right_cumulative, dim=0)

    preintegrated_measurements = {
        'Delta_R_ij': Delta_R_iks[-1],
        'Delta_v_ij': Delta_v_iks[-1],
        'Delta_p_ij': Delta_p_ij,
        'J_R_g_ij': J_R_g_iks[-1],
        'J_v_a_ij': J_v_a_iks[-1],
        'J_v_g_ij': J_v_g_iks[-1],
        'J_p_a_ij': J_p_a_ij,
        'J_p_g_ij': J_p_g_ij,
        'Sigma_ij': Sigma_ij}

    return preintegrated_measurements

class IMUInitializationError(Exception):
    pass

def estimate_gyroscope_bias_iteratively(
    rotations_camera_to_world: torch.Tensor,
    rotation_imu_to_camera: torch.Tensor,
    preintegrated_measurements: Dict[str, Union[pp.LieTensor, torch.Tensor]],
) -> torch.Tensor:
    """Estimate the gyroscope bias iteratively.

    The implementation follows the paper:
    D. Zuñiga-Noël, et al., "An Analytical Solution to the IMU Initialization Problem for Visual-Inertial Systems.", RAL
    2021
    Many of the variable names have been chosen to match and references to equation numbers in the comments are
    referring to this paper unless otherwise stated.

    Args:
        rotations_camera_to_world: Rotations from the camera frame to the world frame, estimated through visual
            odometry (shape [num_keyframes, 3, 3]).
        rotation_imu_to_camera: Rotation from the IMU frame to the camera frame estimated through calibration
            (shape [3, 3]).
        preintegrated_measurements: Preintegrated measurements computed between each pair of rotations estimated through
            visual odometry (a dictionary, as returned by preintegrate_imu).

    Returns:
        The gyroscope bias estimate (shape [3]).
    """
    class PreintegratedRotationResidual(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias_gyro = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

        def forward(self, Delta_R_ijs_vo, Delta_R_ijs_preint, J_R_g_ijs):
            # Compute the updated preintegrated rotation measurement (equation 44 in the Forster paper mentioned in the
            # preintegrate_imu docstring).
            Delta_R_ijs_preint_updated = Delta_R_ijs_preint * pp.so3(J_R_g_ijs @ self.bias_gyro).Exp()

            # Compute the residual (equation 14).
            error = Delta_R_ijs_preint_updated.Inv() * Delta_R_ijs_vo
            return error.Log().tensor()

    # Compute the rotations from the IMU to the world using the camera rotations estimated through visual odometry
    # and the calibrated camera-IMU extrinsics (equation 9).
    R_wbs_vo = pp.from_matrix(rotations_camera_to_world @ rotation_imu_to_camera.unsqueeze(0), ltype=pp.SO3_type,
        check=False)

    # Compute the rotation deltas.
    Delta_R_ijs_vo = R_wbs_vo[:-1].Inv() * R_wbs_vo[1:]

    # Compute and stack the information matrices for each preintegrated rotation measurement.
    Sigma_ijs_inv = torch.stack([pm['Sigma_ij'][:3, :3].inverse() for pm in preintegrated_measurements])

    # Stack the preintegrated rotation measurements.
    Delta_R_ijs_preint = torch.stack([pm['Delta_R_ij'] for pm in preintegrated_measurements])

    # Stack the Jacobians of the preintegrated rotation measurements with respect to the gyroscope bias.
    J_R_g_ijs = torch.stack([pm['J_R_g_ij'] for pm in preintegrated_measurements])

    # Set up the Levenberg-Marquardt optimizer.
    preintegrated_rotation_residual = PreintegratedRotationResidual().to(dtype=J_R_g_ijs.dtype, device=J_R_g_ijs.device)
    optimizer = pp.optim.LM(
        model=preintegrated_rotation_residual,
        weight=Sigma_ijs_inv)
    scheduler = pp.optim.scheduler.StopOnPlateau(
        optimizer=optimizer,
        steps=50, # Default max number of steps in Ceres.
        patience=1)

    # Run the optimization.
    scheduler.optimize(input=(Delta_R_ijs_vo, Delta_R_ijs_preint, J_R_g_ijs))

    return preintegrated_rotation_residual.bias_gyro.data

def estimate_accelerometer_params_analytically(
    transformations_camera_to_world: torch.Tensor,
    timestamps_camera: torch.Tensor,
    transformation_imu_to_camera: torch.Tensor,
    preintegrated_measurements: Dict[str, Union[pp.LieTensor, torch.Tensor]],
    bias_gyro: torch.Tensor,
    bias_accel_precision_prior: float,
    gravity_magnitude: float,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate the accelerometer bias, gravity direction, scale factor, and keyframe velocities analytically.

    The implementation follows the paper:
    D. Zuñiga-Noël, et al., "An Analytical Solution to the IMU Initialization Problem for Visual-Inertial Systems.", RAL
    2021
    to estimate the accelerometer bias, gravity direction, and scale factor. Many of the variable names have been
    chosen to match and references to equation numbers in the comments are referring to this paper unless otherwise
    stated.

    Afterwards, the IMU velocities at each keyframe are computed using the method described in:
    R. Mur-Artal, et al., "Visual-Inertial Monocular SLAM with Map Reuse.", RAL 2017

    Args:
        transformations_camera_to_world: Transformation matrices from the camera frame to the world frame (shape
            [num_keyframes, 4, 4]).
        timestamps_camera: Timestamps (in seconds) corresponding to the transformation matrices (shape [num_keyframes]).
        transformation_imu_to_camera: A transformation matrix from the IMU frame to the camera frame (shape [4, 4]).
        preintegrated_measurements: Preintegrated measurements computed between each pair of rotations estimated through
            visual odometry (a dictionary, as returned by preintegrate_imu).
        bias_gyro: Gyroscope bias estimate (shape [3]).
        bias_accel_precision_prior: Precision (inverse variance, in seconds^4/m^2) of the accelerometer bias prior,
            which is set to zero.
        gravity_magnitude: Magnitude of the acceleration of gravity (in meters/second^2).

    Returns:
        A tuple containing:
            scale_factor: Estimated scale factor that takes the keyframe poses to absolute scale.
            bias_accel: Estimated accelerometer bias (in meters/second^2) (shape [3]).
            gravity_dir_world: Estimated gravity direction in the world frame (a unit vector) (shape [3]).
            velocities_imu: Estimated IMU velocities at each keyframe (in meters/second) (shape [num_keyframes, 3]).
    """
    device = transformations_camera_to_world.device
    dtype = transformations_camera_to_world.dtype

    # Compute the time deltas.
    Delta_ts = (timestamps_camera[1:] - timestamps_camera[:-1]).to(dtype=dtype).unsqueeze(-1).unsqueeze(-1)

    # Set the rotations from the camera to the world estimated through visual odometry.
    R_wcs_vo = transformations_camera_to_world[:, :3, :3]

    # Set the camera positions in the world estimated through visual odometry.
    p_wcs_vo = transformations_camera_to_world[:, :3, 3].unsqueeze(-1)

    # Compute the rotations from the IMU to the world using the camera rotations estimated through visual odometry
    # and the calibrated camera-IMU extrinsics (equation 9).
    R_wbs_vo = transformations_camera_to_world[:, :3, :3] @ transformation_imu_to_camera[:3, :3].unsqueeze(0)

    # Set the position of the IMU in the camera frame from the calibrated camera-IMU extrinsics.
    p_cb = transformation_imu_to_camera[:3, 3].unsqueeze(0).unsqueeze(-1)

    # Stack the needed preintegrated quantities.
    Delta_v_ijs_preint = torch.stack([pm['Delta_v_ij'] for pm in preintegrated_measurements])
    Delta_p_ijs_preint = torch.stack([pm['Delta_p_ij'] for pm in preintegrated_measurements])
    J_v_a_ijs = torch.stack([pm['J_v_a_ij'] for pm in preintegrated_measurements])
    J_p_a_ijs = torch.stack([pm['J_p_a_ij'] for pm in preintegrated_measurements])
    J_v_g_ijs = torch.stack([pm['J_v_g_ij'] for pm in preintegrated_measurements])
    J_p_g_ijs = torch.stack([pm['J_p_g_ij'] for pm in preintegrated_measurements])
    Sigma_ijs = torch.stack([pm['Sigma_ij'] for pm in preintegrated_measurements])

    # Compute the updated preintegrated velocity and position measurements (equation 44 in the Forster paper mentioned
    # in the preintegrate_imu docstring).
    Delta_v_ijs_preint_updated = (Delta_v_ijs_preint + J_v_g_ijs @ bias_gyro).unsqueeze(-1)
    Delta_p_ijs_preint_updated = (Delta_p_ijs_preint + J_p_g_ijs @ bias_gyro).unsqueeze(-1)

    # Compute each A_k, B_k, alpha_k, and pi_k (equations 21-24).
    A_ks = \
        R_wbs_vo[:-2] @ J_p_a_ijs[:-1] / Delta_ts[:-1] - \
        R_wbs_vo[1:-1] @ J_p_a_ijs[1:] / Delta_ts[1:] - \
        R_wbs_vo[:-2] @ J_v_a_ijs[:-1]
    B_ks = -0.5 * (Delta_ts[:-1] + Delta_ts[1:]) * torch.eye(3, dtype=dtype, device=device).repeat(len(A_ks), 1, 1)
    alpha_ks = \
        (p_wcs_vo[2:] - p_wcs_vo[1:-1]) / Delta_ts[1:] - \
        (p_wcs_vo[1:-1] - p_wcs_vo[:-2]) / Delta_ts[:-1]
    pi_ks = \
        R_wbs_vo[1:-1] @ Delta_p_ijs_preint_updated[1:] / Delta_ts[1:] - \
        R_wbs_vo[:-2] @ Delta_p_ijs_preint_updated[:-1] / Delta_ts[:-1] + \
        R_wbs_vo[:-2] @ Delta_v_ijs_preint_updated[:-1] + \
        (R_wcs_vo[1:-1] - R_wcs_vo[:-2]) @ p_cb / Delta_ts[:-1] - \
        (R_wcs_vo[2:] - R_wcs_vo[1:-1]) @ p_cb / Delta_ts[1:]

    # Form each M_k (equation 27).
    M_ks = torch.cat([alpha_ks, A_ks, B_ks], dim=2)

    # Set the information matrix for the prior (equation 18).
    Sigma_inv_prior = torch.zeros(7, 7, dtype=dtype, device=device)
    Sigma_inv_prior[1:4, 1:4] = bias_accel_precision_prior * torch.eye(3, dtype=dtype, device=device)

    # Compute the remaining information matrices.
    # TODO: should R_wcs_vo be R_wbs_vo here? Currently, this matches the original implementation of
    # D. Zuñiga-Noël, et al., but I might want to double check this.
    Sigma_ks = \
        R_wbs_vo[:-2] @ Sigma_ijs[:-1][:, 6:, 6:] @ R_wbs_vo[:-2].mT / Delta_ts[:-1]**2 + \
        R_wbs_vo[1:-1] @ Sigma_ijs[1:][:, 6:, 6:] @ R_wbs_vo[1:-1].mT/ Delta_ts[1:]**2 + \
        R_wcs_vo[:-2] @ Sigma_ijs[:-1][:, 3:6, 3:6] @ R_wcs_vo[:-2].mT
    Sigma_inv_ks = Sigma_ks.inverse()

    # Compute M, m, and Q (equation 27).
    M = Sigma_inv_prior + torch.sum(M_ks.mT @ Sigma_inv_ks @ M_ks, dim=0)
    m = torch.sum(-2 * M_ks.mT @ Sigma_inv_ks @ pi_ks, dim=0)
    Q = torch.sum(pi_ks.mT @ Sigma_inv_ks @ pi_ks, dim=0)

    # Compute A, B, D (equation 34).
    A = 2 * M[:4, :4]
    B = 2 * M[:4, 4:]
    D = 2 * M[4:, 4:]

    # Compute S (introduced below equation 35).
    S = D - B.T @ A.inverse() @ B

    # Compute S^A (equation 40).
    S_A = S.det() * S.inverse()

    # Compute U (equation 43).
    U = S.trace() * torch.eye(3, dtype=dtype, device=device) - S

    # Compute X and Y (equation 45).
    X = 2 * S_A + U @ U
    Y = S_A @ U + U @ S_A

    # Compute the coefficients of lambda on the left side of equation 44.
    mat = torch.zeros(7, 7, dtype=dtype, device=device)
    mat[:4, :4] = A.inverse() @ B @ B.T @ A.inverse().T
    mat[:4, 4:] = -A.inverse() @ B
    mat[4:, :4] = mat[:4, 4:].T
    mat[4:, 4:] = torch.eye(3, dtype=dtype, device=device)
    c4 = (16 * m.T @ mat @ m).item()

    mat[:4, :4] = A.inverse() @ B @ U @ B.T @ A.inverse().T
    mat[:4, 4:] = -A.inverse() @ B @ U
    mat[4:, :4] = mat[:4, 4:].T
    mat[4:, 4:] = U
    c3 = (16 * m.T @ mat @ m).item()

    mat[:4, :4] = A.inverse() @ B @ X @ B.T @ A.inverse().T
    mat[:4, 4:] = -A.inverse() @ B @ X
    mat[4:, :4] = mat[:4, 4:].T
    mat[4:, 4:] = X
    c2 = (4 * m.T @ mat @ m).item()

    mat[:4, :4] = A.inverse() @ B @ Y @ B.T @ A.inverse().T
    mat[:4, 4:] = -A.inverse() @ B @ Y
    mat[4:, :4] = mat[:4, 4:].T
    mat[4:, 4:] = Y
    c1 = (2 * m.T @ mat @ m).item()

    mat[:4, :4] = A.inverse() @ B @ S_A @ S_A @ B.T @ A.inverse().T
    mat[:4, 4:] = -A.inverse() @ B @ S_A @ S_A
    mat[4:, :4] = mat[:4, 4:].T
    mat[4:, 4:] = S_A @ S_A
    c0 = (m.T @ mat @ m).item()

    left_coeffs = np.array([0.0, 0.0, c4, c3, c2, c1, c0])

    # Compute the coefficients of lambda on the right side of equation 44.
    #
    # p(\lambda) = det(S + 2 \lambda I) where I \in \mathbb{R}^{3 \times 3} (equation 40).
    # numpy.poly(A) gives the coefficients of the characteristic polynomial p(t) = det(tI - A)
    # p(\lambda) = det(S + 2 \lambda I) =
    #   det(2(S / 2 + \lambda I)) =
    #   2^3 det(S/2 + lambda I) =
    #   2^3 det(\lambda I - (-S/2))
    # So the coefficients of p(\lambda) are 8 * numpy.poly(-S/2)
    #
    # The right side of equation 44 is: G^2 p(\lambda)^2 (where G is the magnitude of gravity).
    # The coefficients of p(\lambda)^2 are then obtained by numpy.polymul(8 * np.poly(-S / 2), 8 * np.poly(-S / 2))
    # These coefficients are then multiplied by G^2 to get the coefficients on the right side of equation 44.
    p_lambda_coeffs = 8 * np.poly(-S.detach().cpu().numpy() / 2)
    right_coeffs = gravity_magnitude**2 * np.polymul(p_lambda_coeffs, p_lambda_coeffs)

    # Solve for the real roots of the polynomial.
    coeffs = left_coeffs - right_coeffs
    if np.any(np.isnan(coeffs)):
        raise IMUInitializationError('Got a NaN polynomial coefficient. This may be resolved by converting all input ' \
            'tensors to double precision, increasing the noise densities, or lowering the accelerometer bias prior (' \
            'with varying impacts to the results).')
    roots = np.roots(coeffs)
    lambdas = roots[np.isreal(roots)].real
    lambdas = torch.tensor(lambdas, dtype=dtype, device=device)

    # Construct W (equation 28).
    W = torch.eye(7, dtype=dtype, device=device)
    W[:4, :4] = 0.0

    # Compute solutions for each real root (equation 32).
    xs = -(2 * M.unsqueeze(0) + 2 * lambdas.unsqueeze(-1).unsqueeze(-1) * W.unsqueeze(0)).inverse().mT @ m

    # Compute the cost of each solution (equation 26) and select the solution with the minimum cost.
    Cs = xs.mT @ M.unsqueeze(0) @ xs + m.T.unsqueeze(0) @ xs + Q
    x = xs[Cs.argmin()].squeeze()

    # Unpack the solution.
    scale_factor = x[0].item()
    bias_accel = x[1:4]
    gravity_world = x[4:]
    gravity_dir_world = gravity_world / gravity_world.norm()

    # Compute the updated preintegrated velocity and position measurements (equation 44 in the Forster paper mentioned
    # in the preintegrate_imu docstring).
    Delta_v_ijs_preint_updated += (J_v_a_ijs @ bias_accel).unsqueeze(-1)
    Delta_p_ijs_preint_updated += (J_p_a_ijs @ bias_accel).unsqueeze(-1)

    # Compute velocities for all but the last keyframe (equation 18 in the Mur-Artal paper, rearranged to solve for
    # velocity).
    velocities_imu = (
        -0.5 * gravity_world.unsqueeze(0).unsqueeze(-1) * Delta_ts**2 - \
        R_wbs_vo[:-1] @ Delta_p_ijs_preint_updated - \
        (R_wcs_vo[:-1] - R_wcs_vo[1:]) @ p_cb - \
        (p_wcs_vo[:-1] - p_wcs_vo[1:]) * scale_factor) / Delta_ts

    # Compute the velocity for the last keyframe (equation 3 in the Mur-Artal paper).
    velocity_last_keyframe = velocities_imu[-1] + gravity_world.unsqueeze(0).unsqueeze(-1) * Delta_ts[-1] + \
        R_wbs_vo[-1] @ Delta_v_ijs_preint_updated[-1]
    velocities_imu = torch.cat([velocities_imu, velocity_last_keyframe]).squeeze()

    return scale_factor, bias_accel, gravity_dir_world, velocities_imu

def analytical_imu_initialization(
    transformations_camera_to_world: torch.Tensor,
    timestamps_camera: torch.Tensor,
    transformation_imu_to_camera: torch.Tensor,
    linear_accels_meas_imu: torch.Tensor,
    angular_vels_meas_imu: torch.Tensor,
    timestamps_imu: torch.Tensor,
    accelerometer_noise_density: float,
    gyroscope_noise_density: float,
    bias_accel_precision_prior: float = 1e5,
    gravity_magnitude: float = GRAVITY_MAGNITUDE,
) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform (partially) analytical IMU initialization.

    The implementation follows the paper:
    D. Zuñiga-Noël, et al., "An Analytical Solution to the IMU Initialization Problem for Visual-Inertial Systems.", RAL
    2021
    The method first estimates the gyroscope bias iteratively, and then estimates the remaining parameters
    (accelerometer bias, gravity direction, and scale factor) analytically.

    Afterwards, the IMU velocities at each keyframe are computed using the method described in:
    R. Mur-Artal, et al., "Visual-Inertial Monocular SLAM with Map Reuse.", RAL 2017

    Args:
        transformations_camera_to_world: Transformation matrices from the camera frame to the world frame (shape
            [num_keyframes, 4, 4]).
        timestamps_camera: Timestamps (in seconds) corresponding to the transformation matrices (shape [num_keyframes]).
        transformation_imu_to_camera: A transformation matrix from the IMU frame to the camera frame (shape [4, 4]).
        linear_accels_meas_imu: Accelerometer measurements (in meters/second^2) (shape [num_imu_measurements, 3]).
        angular_vels_meas_imu: Gyroscope measurements (in radians/second) (shape [num_imu_measurements, 3]).
        timestamps_imu: IMU measurement timestamps (in seconds) (shape [num_imu_measurements]).
        accelerometer_noise_density: Continuous time accelerometer noise density (in meters/(second^2 sqrt(Hz)) or,
            equivalently, meters/second^1.5).
        gyroscope_noise_density: Continuous time gyroscope noise density (in radians/(second sqrt(Hz)) or,
            equivalently, radians/sqrt(second)).
        bias_accel_precision_prior: Precision (inverse variance, in seconds^4/m^2) of the accelerometer bias prior,
            which is set to zero. Setting this precision prior to a large number essentially holds the accelerometer
            bias close to zero. The paper recommends setting this to a large value for short initialization times
            (the timespan of the keyframes) because over short periods the accelerometer bias is unobservable. However,
            for long initialization times, a large value for this prior may harm the estimation of the scale factor and
            gravity direction.
        gravity_magnitude: Magnitude of the acceleration of gravity (in meters/second^2).

    Returns:
        A tuple containing:
            bias_gyro: Estimated gyroscope bias (in radians/second) (shape [3]).
            scale_factor: Estimated scale factor that takes the keyframe poses to absolute scale.
            bias_accel: Estimated accelerometer bias (in meters/second^2) (shape [3]).
            gravity_dir_world: Estimated gravity direction in the world frame (a unit vector) (shape [3]).
            velocities_imu: Estimated velocities (in meters/second) at each keyframe. The velocities are of the IMU
                frame, with respect to the world frame, expressed in the IMU frame (shape [num_keyframes, 3]).
    """
    # Preintegrate the IMU measurements between each keyframe.
    preintegrated_measurements = []
    for i in range(len(timestamps_camera) - 1):
        # Set the keyframe timestamps.
        timestamp_camera_i = timestamps_camera[i]
        timestamp_camera_j = timestamps_camera[i + 1]

        # Determine the IMU measurements made between the keyframe timestamps.
        interframe_imu_mask = (timestamps_imu >= timestamp_camera_i) & (timestamps_imu < (timestamp_camera_j - 1e-4))

        # If needed, linearly interpolate an IMU measurement at the timestamp of the first keyframe in the pair.
        index_imu_first = torch.where(interframe_imu_mask)[0][0]
        if torch.abs(timestamps_imu[index_imu_first] - timestamp_camera_i) > 1e-4:
            if index_imu_first == 0:
                # There are no earlier IMU measurements. We assume this has happened because the first keyframe is the
                # first processed frame and the IMU data is only available after the first processed frame. In this
                # case, there is likely only a small delay between the first frame/keyframe and the first IMU
                # measurement. Therefore, we simply duplicate the first IMU measurement.
                interpolated_linear_accel = linear_accels_meas_imu[0]
                interpolated_angular_vel = angular_vels_meas_imu[0]
            else:
                timestamp_imu_before = timestamps_imu[index_imu_first - 1]
                timestamp_imu_after = timestamps_imu[index_imu_first]

                linear_accel_before = linear_accels_meas_imu[index_imu_first - 1]
                linear_accel_after = linear_accels_meas_imu[index_imu_first]

                interpolated_linear_accel = linear_accel_before + \
                    (linear_accel_after - linear_accel_before) * (timestamp_camera_i - timestamp_imu_before) / \
                    (timestamp_imu_after - timestamp_imu_before)

                angular_vel_before = angular_vels_meas_imu[index_imu_first - 1]
                angular_vel_after = angular_vels_meas_imu[index_imu_first]

                interpolated_angular_vel = angular_vel_before + \
                    (angular_vel_after - angular_vel_before) * (timestamp_camera_i - timestamp_imu_before) / \
                    (timestamp_imu_after - timestamp_imu_before)

            interframe_linear_accels_meas_imu = \
                torch.cat([interpolated_linear_accel.unsqueeze(0), linear_accels_meas_imu[interframe_imu_mask]])
            interframe_angular_vels_meas_imu = \
                torch.cat([interpolated_angular_vel.unsqueeze(0), angular_vels_meas_imu[interframe_imu_mask]])
            interframe_timestamps_imu = \
                torch.cat([timestamp_camera_i.unsqueeze(0), timestamps_imu[interframe_imu_mask]])
        else:
            # An IMU measurement aligns closely with the timestamp of the first keyframe in the pair, no interpolation
            # is needed.
            interframe_linear_accels_meas_imu = linear_accels_meas_imu[interframe_imu_mask]
            interframe_angular_vels_meas_imu = angular_vels_meas_imu[interframe_imu_mask]
            interframe_timestamps_imu = timestamps_imu[interframe_imu_mask]

        # Preintegrate the IMU measurements.
        preintegrated_measurements.append(preintegrate_imu(
            linear_accels_meas_imu=interframe_linear_accels_meas_imu,
            angular_vels_meas_imu=interframe_angular_vels_meas_imu,
            timestamps_imu=interframe_timestamps_imu,
            t_j=timestamp_camera_j,
            accelerometer_noise_density=accelerometer_noise_density,
            gyroscope_noise_density=gyroscope_noise_density))

    # Estimate the gyroscope bias.
    bias_gyro = estimate_gyroscope_bias_iteratively(
        rotations_camera_to_world=transformations_camera_to_world[:, :3, :3],
        rotation_imu_to_camera=transformation_imu_to_camera[:3, :3],
        preintegrated_measurements=preintegrated_measurements)

    # Estimate the accelerometer bias, gravity direction, and scale factor.
    scale_factor, bias_accel, gravity_dir_world, velocities_imu = estimate_accelerometer_params_analytically(
        transformations_camera_to_world=transformations_camera_to_world,
        timestamps_camera=timestamps_camera,
        transformation_imu_to_camera=transformation_imu_to_camera,
        preintegrated_measurements=preintegrated_measurements,
        bias_gyro=bias_gyro,
        bias_accel_precision_prior=bias_accel_precision_prior,
        gravity_magnitude=gravity_magnitude)

    return bias_gyro, scale_factor, bias_accel, gravity_dir_world, velocities_imu

def integrate_imu(
    times_integrate: torch.Tensor,
    init_rotation: lt.SO3,
    init_linear_velocity: torch.Tensor,
    init_position: torch.Tensor,
    init_time: torch.Tensor,
    linear_accels_meas_imu: torch.Tensor,
    angular_vels_meas_imu: torch.Tensor,
    timestamps_imu: torch.Tensor,
    bias_accel: torch.Tensor,
    bias_gyro: torch.Tensor,
    gravity_dir_world: torch.Tensor,
    gravity_magnitude: float = GRAVITY_MAGNITUDE,
) -> Tuple[torch.Tensor, lt.SO3]:
    # Select the IMU data needed for integration.
    sel_imu_mask = (timestamps_imu >= init_time) & (timestamps_imu < (times_integrate[-1] - 1e-4))
    idx_imu_first = torch.where(sel_imu_mask)[0][0]
    if torch.abs(timestamps_imu[idx_imu_first] - init_time) > 1e-4:
        # Linearly interpolate an IMU measurement at the initial timestamp.
        interp_factor = (init_time - timestamps_imu[idx_imu_first - 1]) / \
            (timestamps_imu[idx_imu_first] - timestamps_imu[idx_imu_first - 1])

        interp_linear_accel = linear_accels_meas_imu[idx_imu_first - 1] + \
            (linear_accels_meas_imu[idx_imu_first] - linear_accels_meas_imu[idx_imu_first - 1]) * interp_factor

        interp_angular_vel = angular_vels_meas_imu[idx_imu_first - 1] + \
            (angular_vels_meas_imu[idx_imu_first] - angular_vels_meas_imu[idx_imu_first - 1]) * interp_factor

        sel_linear_accels_meas_imu = \
            torch.cat([interp_linear_accel.unsqueeze(0), linear_accels_meas_imu[sel_imu_mask]])
        sel_angular_vels_meas_imu = \
            torch.cat([interp_angular_vel.unsqueeze(0), angular_vels_meas_imu[sel_imu_mask]])
        sel_timestamps_imu = \
            torch.cat([init_time, timestamps_imu[sel_imu_mask]])
    else:
        # An IMU measurement aligns closely with the initial timestamp, no interpolation is needed.
        sel_linear_accels_meas_imu = linear_accels_meas_imu[sel_imu_mask]
        sel_angular_vels_meas_imu = angular_vels_meas_imu[sel_imu_mask]
        sel_timestamps_imu = timestamps_imu[sel_imu_mask]
        sel_timestamps_imu[0] = init_time

    # Correct biases.
    unbiased_linear_accels = sel_linear_accels_meas_imu - bias_accel
    unbiased_angular_vels = sel_angular_vels_meas_imu - bias_gyro

    # Compute the gravity vector in the world frame.
    gravity_world = gravity_dir_world * gravity_magnitude

    # Integrate the IMU measurements and retain the integrated rotation, velocity, and position at each IMU
    # measurement. This follows equation 31 in:
    # C. Forster, et al., "On-Manifold Preintegration for Real-Time Visual–Inertial Odometry.", TRO 2017
    Delta_ts = (sel_timestamps_imu[1:] - sel_timestamps_imu[:-1]).to(dtype=torch.double).unsqueeze(-1)
    R_ks = lt.SO3(torch.cat([
        init_rotation.data,
        lt.SO3.exp(unbiased_angular_vels[:-1] * Delta_ts).data]))
    # Cumulative composition of rotations
    for i in torch.pow(2, torch.arange(math.log2(R_ks.shape[0])+1, device=R_ks.device, dtype=torch.int64)):
        if i >= R_ks.shape[0]:
            break
        index = torch.arange(i, R_ks.shape[0], device=R_ks.device, dtype=torch.int64)
        R_ks[index] = (R_ks[index-i] * R_ks[index]).data

    v_ks = torch.cat([
        init_linear_velocity,
        gravity_world * Delta_ts + (R_ks[:-1] * unbiased_linear_accels[:-1]) * Delta_ts]).cumsum_(dim=0)

    p_ks = torch.cat([
        init_position,
        v_ks[:-1] * Delta_ts + 0.5 * gravity_world * Delta_ts**2 + \
            0.5 * (R_ks[:-1] * unbiased_linear_accels[:-1]) * Delta_ts**2]).cumsum_(dim=0)

    # For each time to integrate up to, determine the index of the selected IMU measurement that precedes it.
    idx_prev = torch.searchsorted(sel_timestamps_imu, times_integrate, right=True) - 1

    # Re-integrate the positions and rotations over the sub-intervals between the IMU measurements and the times to
    # integrate up to.
    Delta_ts = (times_integrate - sel_timestamps_imu[idx_prev]).unsqueeze(-1)
    integrated_positions = p_ks[idx_prev] + v_ks[idx_prev] * Delta_ts + 0.5 * gravity_world * Delta_ts**2 + \
        0.5 * (R_ks[idx_prev] * unbiased_linear_accels[idx_prev]) * Delta_ts**2
    integrated_rotations = R_ks[idx_prev] * lt.SO3.exp(unbiased_angular_vels[idx_prev] * Delta_ts)

    return integrated_positions, integrated_rotations
