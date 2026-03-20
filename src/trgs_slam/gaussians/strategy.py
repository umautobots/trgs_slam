from typing import Any, Dict, Union
from dataclasses import dataclass

import torch

from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import reset_opa, _update_param_with_optimizer

from trgs_slam.tools.nested_timer import NestedTimer

@torch.no_grad()
def reset_opa_with_mask(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    value: float,
    mask: torch.Tensor,
) -> None:
    sel = torch.where(mask)[0]

    def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
        if name == 'opacities':
            p[sel] = torch.clamp(p[sel], max=torch.logit(torch.tensor(value)).item())
            return p
        else:
            raise ValueError(f'Unexpected parameter name: {name}')

    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        v[sel] = 0
        return v

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=['opacities']
    )

@dataclass
class SLAMStrategy(DefaultStrategy):
    """Densification strategy for SLAM.

    This is a modification of the gsplat DefaultStrategy that primarily:
    - Adds the option to disable global opacity resetting.
    - Adds the option to perform view-diversity-based opacity resetting.
    - Implements a modified gradient accumulation scheme to account for our microbolometer-aware rendering.

    The argument documentation has been fully rewritten below to remove mentioned defaults (as some have been modified)
    and to provide additional notes or adapt the notes to the SLAM context.

    Args:
        prune_opa: Gaussians with opacity below this value will be pruned.
        grow_grad2d: Gaussians with image plane gradient above this value will be split/duplicated according to
            grow_scale3d.
        grow_scale3d: For Gaussians with image plane gradient above grow_grad2d, those with 3d scale below this value
            will be duplicated and above will be split.
        grow_scale2d: Gaussians with 2d scale (normalized by image resolution) above this value will be split. Ignored
            if refine_scale2d_stop_iter == 0.
        prune_scale3d: Gaussians with 3d scale above this value will be pruned. This is not applied until the iteration
            is greater than reset_every.
        prune_scale2d: Gaussians with 2d scale (normalized by image resolution) above this value will be pruned. This is
            not applied until the iteration is greater than reset_every. Ignored if refine_scale2d_stop_iter == 0.
        refine_scale2d_stop_iter: Stop refining Gaussians based on 2d scale *at* this iteration. Set to a positive value
            to enable grow_scale2d and prune_scale2d to be used.
        refine_start_iter: Start refining Gaussians *after* this iteration. Note that refinement will not happen *at*
            this iteration (even if refine_start_iter % refine_every == 0).
        refine_stop_iter: Stop refining Gaussians *at* this iteration. Note that refinement will not happen *at* this
            iteration (even if refine_stop_iter % refine_every == 0).
        reset_every: Reset opacities every this many iterations. Note that prune_scale3d and prune_scale2d are not
            applied until after this iteration. So, if disable_global_reset is True, reset_every should be set to 1 to
            ensure it does not prevent scale based pruning.
        refine_every: Refine Gaussians every this many iterations.
        pause_refine_after_reset: Pause refining Gaussians until this number of iterations after reset.
        absgrad: Use absolute gradients for Gaussian splitting (following arXiv:2404.10484). If enabled, grow_grad2d
            should be set to a higher value (e.g., 0.0008).
        revised_opacity: Whether to use the revised opacity heuristic from arXiv:2404.06109 (experimental).
        verbose: Whether to print verbose information.
        key_for_gradient: This argument is inherited from the DefaultStrategy and must remain set to "means2d."
        disable_global_reset: Whether to disable global opacity resetting.
        disable_vdb_reset: Whether to disable view-diversity-based opacity resetting.
        condition_num_thresh: If disable_vdb_reset == False, a second moment matrix of view directions will be
            accumulated for each Gaussian and the condition numbers of these matrices will be computed during
            densification. If the log of a condition number is above this threshold the corresponding Gaussian's opacity
            will be reset.
    """
    disable_global_reset: bool = True
    disable_vdb_reset: bool = False
    condition_num_thresh: float = 8.0

    def initialize_state(
        self,
    ) -> Dict[str, Any]:
        state = super().initialize_state()
        if not self.disable_vdb_reset:
            state['V'] = None # Accumulated second moment matrix of Gaussian view directions
        return state

    @NestedTimer.timed_function('Densification')
    @torch.no_grad()
    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        vdb_resetting: bool = False,
        densifying: bool = True,
    ) -> bool:
        densified = False
        if step >= self.refine_stop_iter:
            return densified

        self._update_state(params, state, info)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
            and densifying
        ):
            with NestedTimer('Gaussian Refinement'):
                # Duplicate and split Gaussians.
                n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(params['means'])} GSs."
                    )

                # Prune Gaussians.
                n_prune = self._prune_gs(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(params['means'])} GSs."
                    )

                # Perform view-diversity-based opacity resetting.
                if not self.disable_vdb_reset and state['V'] is not None and vdb_resetting:
                    condition_nums = torch.linalg.cond(state['V'])
                    condition_nums[torch.isnan(condition_nums)] = torch.inf
                    condition_nums = torch.log(condition_nums)
                    reset_mask = condition_nums > self.condition_num_thresh
                    reset_opa_with_mask(params, optimizers, self.prune_opa * 0.5, reset_mask)

                # Reset running states.
                state["grad2d"].zero_()
                state["count"].zero_()
                if not self.disable_vdb_reset and state['V'] is not None:
                    state['V'].zero_()
                if self.refine_scale2d_stop_iter > 0:
                    state["radii"].zero_()
                torch.cuda.empty_cache()

                densified = True

        if (not self.disable_global_reset) and (step != 0) and (step % self.reset_every == 0):
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

        return densified

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> None:
        # Normalize the image plane gradients.
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        if 'integration_weights' in info:
            num_rasters = info['integration_weights'].shape[0]
            grads[..., 0] *= info['width'] / 2.0 * info['n_cameras'] / num_rasters
            grads[..., 1] *= info['height'] / 2.0 * info['n_cameras'] / num_rasters
        else:
            grads[..., 0] *= info['width'] / 2.0 * info['n_cameras']
            grads[..., 1] *= info['height'] / 2.0 * info['n_cameras']

        # Initialize states on the first run.
        n_gaussian = len(list(params.values())[0])
        if state['grad2d'] is None:
            state['grad2d'] = torch.zeros(n_gaussian, device=grads.device)
        if state['count'] is None:
            state['count'] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state['radii'] is None:
            assert 'radii' in info, 'radii is required but missing.'
            state['radii'] = torch.zeros(n_gaussian, device=grads.device)
        if not self.disable_vdb_reset and 'directions' in info and state['V'] is None:
            state['V'] = torch.zeros((n_gaussian, 3, 3), device=grads.device)

        # Update the running states.
        sel = (info['radii'] > 0.0).all(dim=-1)
        camera_ids, gs_ids = torch.where(sel)
        grads = grads[sel]
        radii = info['radii'][sel].max(dim=-1).values
        state['grad2d'].index_add_(0, gs_ids, grads.norm(dim=-1))
        if 'integration_weights' in info:
            # For each raster a Gaussian is visible in, increment the Gaussian's "count" by the value of the weight
            # matrix associated with that raster, evaluated at the Gaussian's projected 2D image coordinate.
            means_2d = info['means2d'][sel]
            x_coords = means_2d[:, 0].round().clamp(0, info['width'] - 1).int()
            y_coords = means_2d[:, 1].round().clamp(0, info['height'] - 1).int()
            weights = info['integration_weights'][camera_ids % num_rasters, y_coords, x_coords]

            state['count'].index_add_(
                0, gs_ids, weights
            )
        else:
            state['count'].index_add_(
                0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
            )
        if self.refine_scale2d_stop_iter > 0:
            state['radii'][gs_ids] = torch.maximum(
                state['radii'][gs_ids],
                radii / float(max(info['width'], info['height'])),
            )

        if not self.disable_vdb_reset and 'directions' in info:
            dirs = info['directions'] # [C, N, 3]
            unit_dirs = dirs / torch.linalg.norm(dirs, dim=2, keepdim=True)
            outer_prods = unit_dirs.unsqueeze(3) @ unit_dirs.unsqueeze(2) # [C, N, 3, 3]
            masked_outer_prods = outer_prods * sel.unsqueeze(-1).unsqueeze(-1) # [C, N, 3, 3]
            summed_outer_prods = masked_outer_prods.sum(dim=0) # [N, 3, 3]
            state['V'] += summed_outer_prods
