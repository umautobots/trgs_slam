from typing import Optional, Literal
from dataclasses import dataclass
import time
import glob
import os

import yaml
import tyro

from trgs_slam.slam import SLAM
from trgs_slam.tools.metrics import ATECalculator

@dataclass
class EvaluationConfig:
    # The directory to load results from.
    load_dir: str
    # The suffix to load.
    load_suffix: Optional[str] = None
    # The alignment method to use in the ATE calculation. When set to 'auto', 'se3' will be used if the trajectory
    # represents the IMU frame (and therefore has a scale estimate), otherwise 'sim3' will be used.
    alignment: Literal['sim3', 'se3', 'auto'] = 'auto'
    # Whether to evaluate the trajectory at keyframe poses only or at all images in the ATE calculation.
    keyframes_only: bool = False
    # Port for the viewer server.
    port: int = 7007

if __name__ == '__main__':
    evaluation_config = tyro.cli(EvaluationConfig, config=(tyro.conf.FlagConversionOff,))

    trial_config_path = f'{evaluation_config.load_dir}/config.yaml'
    with open(trial_config_path, 'r') as file:
        trial_config = yaml.load(file, Loader=yaml.Loader)

    # Find the final checkpoint if the load suffix was not specified.
    if evaluation_config.load_suffix is None:
        gaussians_dir = f'{evaluation_config.load_dir}/gaussians/'
        final_files = glob.glob(os.path.join(gaussians_dir, '*final*.pt'))
        if final_files:
            filename = os.path.basename(final_files[0])
            evaluation_config.load_suffix = filename.replace('gaussians_', '').replace('.pt', '')
            print(f'Defaulting load_suffix to the final checkpoint: {evaluation_config.load_suffix}')
        else:
            raise FileNotFoundError(f"No final checkpoint found in {gaussians_dir} and 'load_suffix' was not "
                "specified.")

    # Load the SLAM result.
    trial_config.load_dir = evaluation_config.load_dir
    trial_config.load_suffix = evaluation_config.load_suffix
    trial_config.viewer_config.port = evaluation_config.port
    trial_config.result_dir = None # Ensure the saved config is not overwritten.
    trial_config.keyframe_config.cache_location = 'disk' # Skip loading keyframe images to CPU or GPU.
    trial_config.viewer_config.disable_viewer = False # Enable the viewer.
    trial_config.viewer_config.mode = 'rendering' # Set to rendering mode to keep the lock released and simplify GUI.
    slam = SLAM(trial_config)
    slam.keyframe_manager.window_indices = [] # Clear the keyframe window -- not relevant outside of SLAM.

    # Visualize the estimated positions.
    slam.viewer.update_viewer_kfs(mode=None, force=True)

    # Compute ATE and plot the ground truth positions.
    ate_calculator = ATECalculator(slam.dataset, slam.trajectory_manager, slam.keyframe_manager)
    if ate_calculator.enabled:
        rmse_ate, percent_scale_error = ate_calculator.compute_rmse_ate(
            evaluation_config.alignment,
            evaluation_config.keyframes_only,
            slam.slam_state.current_index,
            slam.viewer,
            show_gt_traj_init_val=True)
        print(f'RMSE ATE: {rmse_ate}')
        if percent_scale_error is not None:
            print(f'Percent scale error: {percent_scale_error}')

    while True:
        time.sleep(0.1)
