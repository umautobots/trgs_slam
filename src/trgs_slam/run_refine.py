import yaml
import numpy as np
import tyro
import glob
import os

from trgs_slam.refine import RefineConfig, Refine
from trgs_slam.slam import SLAMConfig
from trgs_slam.gaussians.gaussians import GaussianConfig
from trgs_slam.gaussians.strategy import SLAMStrategy
from trgs_slam.tools.nested_timer import NestedTimer

if __name__ == '__main__':
    # Initialize the config with defaults and overwrite with command line values.
    default_strategy = SLAMStrategy(
        prune_opa=0.005,
        refine_start_iter=100,
        refine_stop_iter=4_000,
        refine_every=250,
        reset_every=3_000,
        disable_global_reset=False,
        disable_vdb_reset=True,
        absgrad=False,
        grow_grad2d=0.0002,
        revised_opacity=False)
    default_gaussian_config = GaussianConfig(strategy=default_strategy)

    config = tyro.cli(
        RefineConfig,
        default=RefineConfig(
            slam_config=SLAMConfig(
                load_strategy_state=False,
                load_optimizer_states=False,
                load_freeze_states=False,
                gaussian_config=default_gaussian_config,
            ),
        ),
        config=(tyro.conf.FlagConversionOff,),
    )

    assert config.slam_config.load_dir is not None, 'The load_dir must be specified for refinement.'

    # Load the saved SLAM config.
    path_config = f'{config.slam_config.load_dir}/config.yaml'
    with open(path_config, 'r') as file:
        saved_config = yaml.load(file, Loader=yaml.Loader)

    # Overwrite elements of the saved SLAM config that have different defaults in refinement.
    saved_config.load_strategy_state = False
    saved_config.load_optimizer_states = False
    saved_config.load_freeze_states = False
    saved_config.gaussian_config.strategy = default_strategy

    # Ensure that SLAM results are not overwritten if a new result_dir has not been provided for refinement.
    saved_config.result_dir = None

    # Ignore any saved load suffix.
    saved_config.load_suffix = None

    # Reinitialize the config with the saved settings as default and overwrite with command line values.
    config = tyro.cli(
        RefineConfig,
        default=RefineConfig(
            slam_config=saved_config,
        ),
        config=(tyro.conf.FlagConversionOff,),
    )

    # Find the final checkpoint if the load suffix was not specified.
    if config.slam_config.load_suffix is None:
        gaussians_dir = f'{config.slam_config.load_dir}/gaussians/'
        final_files = glob.glob(os.path.join(gaussians_dir, '*final*.pt'))
        if final_files:
            filename = os.path.basename(final_files[0])
            config.slam_config.load_suffix = filename.replace('gaussians_', '').replace('.pt', '')
            print(f'Defaulting load_suffix to the final checkpoint: {config.slam_config.load_suffix}')
        else:
            raise FileNotFoundError(f"No final checkpoint found in {gaussians_dir} and 'load_suffix' was not "
                "specified.")

    # Configure global timer settings.
    NestedTimer.disable = not config.slam_config.timer_config.enable
    NestedTimer.force_cuda_sync = config.slam_config.timer_config.force_cuda_sync
    if config.slam_config.timer_config.print_mean_only:
        NestedTimer.set_custom_stat_funcs({'mean_ms': lambda x: float(np.mean(x))})

    # Run refinement.
    with NestedTimer('Refine'):
        refine = Refine(config)

    # Save the final result.
    refine.save(suffix_extension='final', save_timing=True)

    # If enabled, leave the viewer running.
    refine.viewer.complete()
