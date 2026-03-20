from typing import Literal, Optional
from dataclasses import dataclass
import time
import collections
import math

import numpy as np
import torch

import viser
from nerfview import Viewer, RenderTabState, CameraState, apply_float_colormap
from nerfview._renderer import RenderTask

from trgs_slam.renderer.renderer import Renderer
from trgs_slam.gaussians.gaussians import GaussianManager
from trgs_slam.trajectory.trajectory import TrajectoryManager
from trgs_slam.trajectory.keyframes import KeyframeManager
from trgs_slam.tools.nested_timer import NestedTimer, nested_timer_paused
from trgs_slam.datasets.base_dataset import BaseDataset

@dataclass
class SLAMViewerConfig:
    # Whether to disable the viewer.
    disable_viewer: bool = False
    # Port for the viewer server.
    port: int = 7007
    # Update the visualization of keyframes and trajectory at every mapping and tracking iteration. This is very slow
    # but is useful for debugging (e.g., tuning the control point learning rates).
    update_traj_each_iter: bool = False
    # Viewer mode. In training mode, the viewer lock is acquired immediately and is only released when training is
    # paused, during post loop updates, and when training is complete. In rendering mode, the lock is not acquired on
    # construction. Training mode also includes an additional tab in the GUI.
    mode: Literal['rendering', 'training'] = 'training'
    # Whether to close the viewer when training (SLAM or refinement) is complete.
    close_on_completion: bool = True

class SLAMRenderTabState(RenderTabState):
    # Statistics.
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # Controllable parameters.
    condition_num_thresh_low = 5
    condition_num_thresh_high = 8
    render_mode: Literal['thermal', 'depth', 'alpha'] = 'thermal'
    show_pos_spline: bool = True
    show_all_kfs: bool = False
    composite_depth: bool = False
    colormap: Literal['turbo', 'viridis', 'magma', 'inferno', 'cividis', 'gray'] = 'turbo'


class SLAMViewer(Viewer):
    def __init__(
        self,
        config: SLAMViewerConfig,
        renderer: Renderer,
        gaussian_manager: GaussianManager,
        trajectory_manager: TrajectoryManager,
        keyframe_manager: KeyframeManager,
        dataset: BaseDataset,
    ) -> None:
        self.config = config

        if self.config.disable_viewer:
            return

        # Set the SLAM components.
        self.renderer = renderer
        self.trajectory_manager = trajectory_manager
        self.keyframe_manager = keyframe_manager
        self.gaussian_manager = gaussian_manager
        self.dataset = dataset

        # Initialize the viser server.
        server = viser.ViserServer(port=self.config.port, verbose=False)
        @server.on_client_connect # Set the camera to align with the pose of the first frame.
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = (0., 0., -0.2)
            client.camera.wxyz = (1., 0., 0., 0.)

        # Run the base class constructor.
        super().__init__(server, self._render_fn, mode=self.config.mode)

        # Customize the GUI.
        server.gui.set_panel_label('SLAM Viewer')
        server.gui.configure_theme(
            control_width='large',
            show_logo=False,
            dark_mode=True,
            show_share_button=False,
            brand_color=(3, 252, 244))

        # If training, setup the training tab.
        if self.config.mode == 'training':
            self._training_tab_handles['step_number'].hint = 'Total number of mapping steps.'
            self._training_tab_handles['train_util_slider'].hint = 'The fraction of execution time dedicated to ' \
                'training (as opposed to visualization).'

        # If training, acquire the lock immediately. For simplicity, the lock is only released when training is paused,
        # during post loop updates, and when training is complete.
        self._lock_acquired = False
        if self.config.mode == 'training':
            self.acquire_lock()

        # Initialize attributes to store viewer elements.
        self._current_image_handle = None
        self._current_render_handle = None
        self._current_fpn_handle = None

        # Compute HFOV and aspect ratio for visualizing camera frustums.
        fx = dataset.new_camera_matrix[0, 0].item()
        self.hfov = 2 * np.arctan(dataset.image_width / (2 * fx))
        self.aspect = dataset.image_width / dataset.image_height

        # Initialize timing variables.
        self._loop_start_time = None
        self._training_times = collections.deque(maxlen=100)
        self._img_and_kf_update_times = collections.deque(maxlen=5)
        self._viewer_render_times = collections.deque(maxlen=5)

    def _init_training_tab(self) -> None:
        self._training_tab_handles = {}
        self._training_folder = self.server.gui.add_folder('Training', order=0)

    def _init_rendering_tab(self) -> None:
        self.render_tab_state = SLAMRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder('Viewer Rendering', order=1)

    def _populate_rendering_tab(self) -> None:
        with self._rendering_folder:
            # Setup the resolution slider.
            viewer_res_slider = self.server.gui.add_slider(
                'Viewer Res.',
                min=64,
                max=2048,
                step=1,
                initial_value=self.render_tab_state.viewer_res,
                hint='Maximum resolution of the viewer rendered image.')

            @viewer_res_slider.on_update
            def _(_) -> None:
                self.render_tab_state.viewer_res = int(viewer_res_slider.value)
                self.rerender(_)

            self._rendering_tab_handles['viewer_res_slider'] = viewer_res_slider

            # Setup the condition number normalization threshold multi slider.
            initial_value = (
                self.render_tab_state.condition_num_thresh_low,
                self.render_tab_state.condition_num_thresh_high)
            condition_num_thresh_multi_slider = self.server.gui.add_multi_slider(
                'Condition Num. Thresholds',
                min=1,
                max=15,
                step=0.2,
                initial_value=initial_value,
                hint='Thresholds used in normalizing the log of the condition numbers for the viewer.')

            @condition_num_thresh_multi_slider.on_update
            def _(_) -> None:
                value = condition_num_thresh_multi_slider.value
                self.render_tab_state.condition_num_thresh_low = value[0]
                self.render_tab_state.condition_num_thresh_high = value[1]
                self.rerender(_)

            self._rendering_tab_handles['condition_num_thresh_multi_slider'] = condition_num_thresh_multi_slider

            # Setup the Gaussian stats.
            total_gs_count_number = self.server.gui.add_number(
                'Total Number of Gaussians',
                initial_value=self.render_tab_state.total_gs_count,
                disabled=True,
                hint='Total number of Gaussians in the scene.')
            self._rendering_tab_handles['total_gs_count_number'] = total_gs_count_number

            rendered_gs_count_number = self.server.gui.add_number(
                'Number in Viewer FOV',
                initial_value=self.render_tab_state.rendered_gs_count,
                disabled=True,
                hint='Number of Gaussians rendered in the current viewer FOV.')
            self._rendering_tab_handles['rendered_gs_count_number'] = rendered_gs_count_number

            # Setup the render mode dropdown.
            render_mode_dropdown = self.server.gui.add_dropdown(
                'Render Mode',
                ('thermal', 'depth', 'alpha', 'condition_num_log'),
                initial_value=self.render_tab_state.render_mode,
                hint='Render mode to use.')

            @render_mode_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.render_mode = render_mode_dropdown.value
                self.rerender(_)

            self._rendering_tab_handles['render_mode_dropdown'] = render_mode_dropdown

            # Setup the show position spline checkbox.
            show_pos_spline_checkbox = self.server.gui.add_checkbox(
                'Show Position Spline',
                initial_value=self.render_tab_state.show_pos_spline,
                hint='Show the position spline.')

            @show_pos_spline_checkbox.on_update
            def _(_) -> None:
                self.render_tab_state.show_pos_spline = show_pos_spline_checkbox.value
                self.acquire_lock()
                self.update_viewer_kfs(mode=None)
                self.release_lock()

            self._rendering_tab_handles['show_pos_spline_checkbox'] = show_pos_spline_checkbox

            # Setup the show all keyframes checkbox.
            show_all_kfs_checkbox = self.server.gui.add_checkbox(
                'Show All KFs',
                initial_value=self.render_tab_state.show_all_kfs,
                hint='Show all keyframes (rather than just those in the current window).')

            @show_all_kfs_checkbox.on_update
            def _(_) -> None:
                self.render_tab_state.show_all_kfs = show_all_kfs_checkbox.value
                self.acquire_lock()
                self.update_viewer_kfs(mode='mapping')
                self.release_lock()

            self._rendering_tab_handles['show_all_kfs_checkbox'] = show_all_kfs_checkbox

            # Setup the composite depth checkbox.
            composite_depth_checkbox = self.server.gui.add_checkbox(
                'Composite Depth',
                initial_value=self.render_tab_state.composite_depth,
                hint='Allow Gaussians to occlude the trajectory and keyframes.')

            @composite_depth_checkbox.on_update
            def _(_) -> None:
                self.render_tab_state.composite_depth = composite_depth_checkbox.value
                self.rerender(_)

            self._rendering_tab_handles['composite_depth_checkbox'] = composite_depth_checkbox

            # Setup the colormap dropdown.
            colormap_dropdown = self.server.gui.add_dropdown(
                'Colormap',
                ('turbo', 'viridis', 'magma', 'inferno', 'cividis', 'gray'),
                initial_value=self.render_tab_state.colormap,
                hint='Colormap used for the depth, alpha, and condition_num_log render modes.')

            @colormap_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.colormap = colormap_dropdown.value
                self.rerender(_)

            self._rendering_tab_handles['colormap_dropdown'] = colormap_dropdown

    def acquire_lock(self) -> None:
        if self.config.disable_viewer:
            return

        if not self._lock_acquired:
            self.lock.acquire()
            self._lock_acquired = True

    def release_lock(self) -> None:
        if self.config.disable_viewer:
            return

        if self._lock_acquired:
            self.lock.release()
            self._lock_acquired = False

    def pre_loop(self) -> None:
        if self.config.disable_viewer:
            return

        if self.state == 'paused':
            with nested_timer_paused(): # Ignore time spent paused.
                self.release_lock()
                while self.state == 'paused':
                    time.sleep(0.01)
                self.acquire_lock()

        self._loop_start_time = time.time()

    @NestedTimer.timed_function('Viewer Post Loop')
    def post_loop(
        self,
        global_mapping_step: Optional[int] = None,
        frame: Optional[torch.Tensor] = None,
        render: Optional[torch.Tensor] = None,
        fpn_time: Optional[torch.Tensor] = None,
    ) -> None:
        if self.config.disable_viewer:
            return

        assert self._loop_start_time is not None, 'Cannot run `post_loop` before `pre_loop`.'

        # Skip if no clients are connected.
        if len(self._renderers) == 0:
            return

        self.release_lock()

        if global_mapping_step is not None:
            self._training_times.append(time.time() - self._loop_start_time)

        with nested_timer_paused(): # Ignore time spent waiting for user inactivity.
            # Pause training while the user moves the camera to make viewing smoother.
            while time.time() - self._last_move_time < 0.1:
                time.sleep(0.05)

        rendering_submitted = False
        if global_mapping_step is not None:
            # Update the global step count in the viewer.
            self._training_tab_handles['step_number'].value = global_mapping_step

            # Compute moving averages of the training and viewer updates times.
            mean_training_time = np.mean(np.array(self._training_times))
            mean_img_and_kf_update_time = np.mean(np.array(self._img_and_kf_update_times)) if \
                len(self._img_and_kf_update_times) != 0 else 0.0
            mean_viewer_render_times = np.mean(np.array(self._viewer_render_times)) if \
                len(self._viewer_render_times) != 0 else 0.0
            mean_viewer_update_time = mean_img_and_kf_update_time + mean_viewer_render_times

            # Submit rendering tasks at a rate that approximately maintains the set training utility (the fraction of
            # the compute time dedicated to training).
            if self.state == 'training' and self._training_tab_handles['train_util_slider'].value != 1:
                train_util = self._training_tab_handles['train_util_slider'].value
                update_every = mean_viewer_update_time / ((1 - train_util) * mean_training_time)
                if global_mapping_step > self._last_update_step + update_every:
                    self._last_update_step = global_mapping_step
                    clients = self.server.get_clients()
                    for client_id in clients:
                        camera_state = self.get_camera_state(clients[client_id])
                        assert camera_state is not None
                        self._renderers[client_id].submit(RenderTask('update', camera_state))
                    rendering_submitted = True

        with NestedTimer('Viewer Lock Acquisition (waits for viewer rendering to complete, if any)'):
            self.acquire_lock()

        img_and_kf_update_start_time = time.time()

        if rendering_submitted:
            self.update_images(frame, render, fpn_time)
        if self.config.update_traj_each_iter or rendering_submitted:
            self.update_viewer_kfs(mode='mapping')

        if rendering_submitted:
            self._img_and_kf_update_times.append(time.time() - img_and_kf_update_start_time)

    @torch.no_grad()
    def _render_fn(
        self,
        camera_state: CameraState,
        render_tab_state: SLAMRenderTabState,
    ) -> None:
        viewer_render_start_time = time.time()

        # Set the perspective of the viewer.
        transformation_world_to_camera = \
            torch.from_numpy(camera_state.c2w).float().to(self.gaussian_manager.config.device).inverse()
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height
        camera_matrix = \
            torch.from_numpy(camera_state.get_K((width, height))).float().to(self.gaussian_manager.config.device)

        # If the 'condition_num_log' render mode is selected, set the Gaussian intensities to the normalized log of
        # their corresponding condition numbers.
        if render_tab_state.render_mode == 'condition_num_log':
            condition_nums = torch.log(torch.linalg.cond(self.gaussian_manager.strategy_state['V']))
            condition_nums[torch.isnan(condition_nums)] = render_tab_state.condition_num_thresh_high
            condition_nums[condition_nums > render_tab_state.condition_num_thresh_high] = \
                render_tab_state.condition_num_thresh_high
            condition_nums[condition_nums < render_tab_state.condition_num_thresh_low] = \
                render_tab_state.condition_num_thresh_low
            condition_nums = \
                (condition_nums - render_tab_state.condition_num_thresh_low) / \
                (render_tab_state.condition_num_thresh_high - render_tab_state.condition_num_thresh_low)
            intensities = condition_nums.unsqueeze(-1)
        else:
            intensities = None

        # Perform rendering.
        if render_tab_state.composite_depth:
            render_mode_map = {'thermal': 'RGB+ED', 'depth': 'ED', 'alpha': 'ED', 'condition_num_log': 'RGB+ED'}
        else:
            render_mode_map = {'thermal': 'RGB', 'depth': 'ED', 'alpha': 'RGB', 'condition_num_log': 'RGB'}
        rendered_image, rendered_alpha, info = self.renderer.rasterize(
            transformations_world_to_camera=transformation_world_to_camera.unsqueeze(0),
            Ks=camera_matrix.unsqueeze(0),
            width=width,
            height=height,
            near_plane=1e-3,
            far_plane=1e3,
            intensities=intensities,
            render_mode=render_mode_map[render_tab_state.render_mode])

        # Set the background image.
        if render_tab_state.render_mode == 'thermal':
            background_image = rendered_image[..., 0].squeeze().clamp(0, 1).cpu().numpy()[:, :, None].repeat(3, axis=2)
        elif render_tab_state.render_mode == 'depth':
            depth = rendered_image[..., -1:].squeeze(0)
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-10)).clamp(0, 1)
            background_image = apply_float_colormap(depth_norm, render_tab_state.colormap).cpu().numpy()
        elif render_tab_state.render_mode == 'alpha':
            background_image = apply_float_colormap(rendered_alpha.squeeze(0), render_tab_state.colormap).cpu().numpy()
        elif render_tab_state.render_mode == 'condition_num_log':
            condition_num_image = rendered_image[..., :1].squeeze(0).clamp(0, 1)
            background_image = apply_float_colormap(condition_num_image, render_tab_state.colormap).cpu().numpy()

        # Update the statistics in the viewer.
        render_tab_state.total_gs_count = len(self.gaussian_manager.params['means'])
        render_tab_state.rendered_gs_count = (info['radii'] > 0).all(-1).sum().item()
        self._rendering_tab_handles['total_gs_count_number'].value = render_tab_state.total_gs_count
        self._rendering_tab_handles['rendered_gs_count_number'].value = render_tab_state.rendered_gs_count

        # If selected, allow the Gaussians to occlude the trajectory and keyframes.
        if render_tab_state.composite_depth:
            depth = rendered_image[..., -1:].squeeze()
            depth[rendered_alpha.squeeze() < 0.1] = 1e3

            # Downsample to avoid slowing down the viewer too much.
            desired_depth_pixels = 512**2
            current_depth_pixels = depth.shape[0] * depth.shape[1]
            scale = min(desired_depth_pixels / max(1, current_depth_pixels), 1.0)
            depth = torch.nn.functional.interpolate(
                depth[None, None, ...],
                size=(int(depth.shape[0] * scale), int(depth.shape[1] * scale)),
                mode='bilinear')[0, 0, :, :].cpu().numpy()

            self._viewer_render_times.append(time.time() - viewer_render_start_time)
            return (background_image, depth)

        self._viewer_render_times.append(time.time() - viewer_render_start_time)
        return background_image

    @NestedTimer.timed_function('Update Viewer Keyframes and Traj.')
    @torch.no_grad()
    def update_viewer_kfs(
        self,
        mode: Optional[Literal['mapping', 'tracking']] = None,
        force: bool = False,
        tracking_timestamp: Optional[float] = None,
    ) -> None:
        if self.config.disable_viewer:
            return

        # Skip if no clients are connected (unless force == True).
        if not force and len(self._renderers) == 0:
            return

        if mode == 'mapping':
            # Read in the keyframe timestamps.
            kf_timestamps = torch.cat([kf.timestamp for kf in self.keyframe_manager.keyframes])

            # Evaluate the keyframe positions and orientations.
            kf_positions_camera_in_world, kf_rotations_camera_to_world = \
                self.trajectory_manager(kf_timestamps, as_matrices=False)
            kf_positions_camera_in_world = kf_positions_camera_in_world.cpu().numpy()
            kf_quats_wxyz_camera_to_world = kf_rotations_camera_to_world.data.cpu().numpy()[:, [3, 0, 1, 2]]

            # Update the keyframe camera frustums in the GUI.
            for i in range(len(kf_timestamps)):
                name = f'kf{i}'
                in_window = i in self.keyframe_manager.window_indices
                color = (0, 255, 0) if in_window else (255, 0, 0)
                camera_handle = self.server.scene.get_handle_by_name(name)
                if in_window or self.render_tab_state.show_all_kfs:
                    if camera_handle is None:
                        self.server.scene.add_camera_frustum(
                            name=name,
                            fov=self.hfov,
                            scale=0.01,
                            color=color,
                            aspect=self.aspect,
                            wxyz=kf_quats_wxyz_camera_to_world[i],
                            position=kf_positions_camera_in_world[i])
                    else:
                        camera_handle.wxyz = kf_quats_wxyz_camera_to_world[i]
                        camera_handle.position = kf_positions_camera_in_world[i]
                        camera_handle.color = color
                elif camera_handle is not None:
                    self.server.scene.remove_by_name(name)

            # Remove the camera frustum created during tracking.
            self.server.scene.remove_by_name('current_frame')

        elif mode == 'tracking' and tracking_timestamp is not None:
            # Update the current camera frustum in the GUI.
            lte_position_camera_in_world, lte_rotation_camera_to_world = \
                self.trajectory_manager(tracking_timestamp, as_matrices=False)
            lte_position_camera_in_world = lte_position_camera_in_world.cpu().numpy()[0]
            lte_rotation_camera_to_world = lte_rotation_camera_to_world.data.cpu().numpy()[0, [3, 0, 1, 2]]
            self.server.scene.add_camera_frustum(
                name='current_frame',
                fov=self.hfov,
                scale=0.01,
                color=(0, 0, 255),
                aspect=self.aspect,
                wxyz=lte_rotation_camera_to_world,
                position=lte_position_camera_in_world)

        # Update the position spline in the GUI.
        pos_spline_handle = self.server.scene.get_handle_by_name('position_spline')
        if self.render_tab_state.show_pos_spline:
            first_time = self.keyframe_manager.keyframes[0].timestamp.item()
            last_time = self.trajectory_manager.latest_time_eval
            line_segment_times = torch.linspace(
                first_time,
                last_time,
                int((last_time - first_time) / self.dataset.average_frame_period) + 2,
                dtype=torch.double,
                device=self.trajectory_manager.config.device)
            ls_positions_camera_in_world, _ = self.trajectory_manager(line_segment_times, as_matrices=False)
            points = \
                torch.stack((ls_positions_camera_in_world[:-1], ls_positions_camera_in_world[1:]), dim=1).cpu().numpy()
            self.server.scene.add_line_segments(
                name='position_spline',
                points=points,
                colors=(0, 0, 255),
                line_width=3)
        elif pos_spline_handle is not None:
            self.server.scene.remove_by_name('position_spline')

    @torch.no_grad()
    def update_images(
        self,
        frame: Optional[torch.Tensor],
        render: Optional[torch.Tensor],
        fpn_time: Optional[torch.Tensor],
    ) -> None:
        if self.config.disable_viewer:
            return

        if frame is not None:
            self.update_frame(frame)
        if render is not None:
            self.update_render(render)
        if fpn_time is not None:
            self.update_fpn(fpn_time)

    @torch.no_grad()
    def update_image(
        self,
        image: torch.Tensor,
        handle: viser.GuiImageHandle,
    ) -> None:
        if self.config.disable_viewer:
            return

        # Skip if no clients are connected.
        if len(self._renderers) == 0:
            return

        # Upsample if needed.
        image = image.squeeze()
        factor = int(math.ceil(512.0 / image.shape[0]))
        if factor > 1:
            new_shape = (image.shape[0] * factor, image.shape[1] * factor)
            image = torch.nn.functional.interpolate(image[None, None, ...], size=new_shape, mode='nearest').squeeze()

        # Set the image in the GUI.
        image = image[:, :, None].repeat(1, 1, 3).cpu().numpy()
        handle.image = image

    @NestedTimer.timed_function('Update Viewer Current Frame')
    @torch.no_grad()
    def update_frame(
        self,
        frame: torch.Tensor,
    ) -> None:
        if self.config.disable_viewer:
            return

        if self._current_image_handle is None:
            image = np.zeros((1, 1, 3))
            self._current_image_handle = \
                self.server.gui.add_image(image, label='Current Frame', format='jpeg', jpeg_quality=95, order=2)
        self.update_image(frame, self._current_image_handle)

    @NestedTimer.timed_function('Update Viewer Current Frame Render')
    @torch.no_grad()
    def update_render(
        self,
        render: torch.Tensor,
    ) -> None:
        if self.config.disable_viewer:
            return

        if self._current_render_handle is None:
            image = np.zeros((1, 1, 3))
            self._current_render_handle = \
                self.server.gui.add_image(image, label='Current Frame Render', format='jpeg', jpeg_quality=95, order=2)
        self.update_image(render, self._current_render_handle)

    @NestedTimer.timed_function('Update Viewer Current FPN')
    @torch.no_grad()
    def update_fpn(
        self,
        fpn_time: torch.Tensor,
    ) -> None:
        if self.config.disable_viewer or not self.renderer.fpn_manager.config.enable:
            return

        fpn = self.renderer.fpn_manager.fpn.pixelwise_offsets_spline(fpn_time)
        fpn = fpn.reshape(self.dataset.image_height, self.dataset.image_width)
        fpn = (fpn - fpn.min()) / (fpn.max() - fpn.min() + 1e-10)

        if self._current_fpn_handle is None:
            image = np.zeros((1, 1, 3))
            self._current_fpn_handle = \
                self.server.gui.add_image(image, label='Current FPN', format='jpeg', jpeg_quality=95, order=2)
        self.update_image(fpn, self._current_fpn_handle)

    def complete(self) -> None:
        if self.config.disable_viewer:
            return

        self.release_lock()
        super().complete()
        if not self.config.close_on_completion:
            print("Viewer running... Ctrl+C to exit.")
            time.sleep(1000000)
