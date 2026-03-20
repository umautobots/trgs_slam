import yaml
import numpy as np
import tyro

from trgs_slam.slam import SLAMConfig, SLAM
from trgs_slam.gaussians.gaussians import GaussianConfig
from trgs_slam.gaussians.strategy import SLAMStrategy
from trgs_slam.tools.nested_timer import NestedTimer

if __name__ == '__main__':
    # Initialize the config with defaults and overwrite with command line values.
    config = tyro.cli(
        SLAMConfig,
        default=SLAMConfig(
            gaussian_config=GaussianConfig(
                strategy=SLAMStrategy(
                    prune_opa=0.01,
                    refine_start_iter=0,
                    refine_stop_iter=float('inf'),
                    refine_every=2000,
                    reset_every=1,
                    absgrad=False,
                    grow_grad2d=0.0004,
                    revised_opacity=False,
                ),
            ),
        ),
        config=(tyro.conf.FlagConversionOff,),
    )

    if config.load_dir is not None:
        assert config.load_suffix, 'Must provide load_suffix if load_dir is not None'

        path_config = f'{config.load_dir}/config.yaml'
        with open(path_config, 'r') as file:
            saved_config = yaml.load(file, Loader=yaml.Loader)

        # Reinitialize the config with the saved settings as default and overwrite with command line values.
        config = tyro.cli(
            SLAMConfig,
            default=saved_config,
            config=(tyro.conf.FlagConversionOff,),
        )

    # Configure global timer settings.
    NestedTimer.disable = not config.timer_config.enable
    NestedTimer.force_cuda_sync = config.timer_config.force_cuda_sync
    if config.timer_config.print_mean_only:
        NestedTimer.set_custom_stat_funcs({'mean_ms': lambda x: float(np.mean(x))})

    # Run SLAM.
    with NestedTimer('SLAM'):
        slam = SLAM(config)
        slam.run()

    # Save the final result.
    slam.save(suffix_extension='final', save_timing=True)

    # If enabled, leave the viewer running.
    slam.viewer.complete()
