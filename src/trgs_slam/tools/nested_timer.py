import os
import time
import pprint
import contextlib
import threading
from pathlib import Path
from functools import wraps

import yaml
import numpy as np
import torch
import h5py

def get_stats(task_times_ms, printed):
    if len(task_times_ms) == 1:
        stats = {
            'num_runs': len(task_times_ms),
            'time_ms': float(task_times_ms[0]),
            'percent': None}
    elif len(task_times_ms) > 1:
        stats = {
            'num_runs': len(task_times_ms),
            'total_time_ms': float(np.sum(task_times_ms)),
            'percent': None}
        stat_funcs = NestedTimer._printed_stat_funcs if printed else NestedTimer._saved_stat_funcs
        for stat_name, stat_func in stat_funcs.items():
            stats[stat_name] = stat_func(task_times_ms)
    else:
        stats = {'num_runs': 0}

    return stats

def set_percent(summary, total_time_ms_parent=0):
    percent_timed = 0
    for stats in summary.values():
        if stats['num_runs'] != 0:
            total_time_ms = stats['time_ms'] if stats['num_runs'] == 1 else stats['total_time_ms']
            if total_time_ms_parent > 0:
                stats['percent'] = (total_time_ms / total_time_ms_parent) * 100
                percent_timed += stats['percent']
            else:
                del stats['percent']

            if 'subtasks' in stats and stats['subtasks']:
                set_percent(stats['subtasks'], total_time_ms)
        else:
            if 'subtasks' in stats and stats['subtasks']:
                set_percent(stats['subtasks'])

    if total_time_ms_parent > 0:
        summary['percent_other_subtasks'] = 100 - percent_timed # Includes the overhead of the NestedTimer itself.

def populate_summary(time_data, summary, run_id_first=-1, run_id_last=-1):
    if not time_data:
        return summary

    for task_name, run_data in time_data.items():
        task_times_ms = np.array(run_data['task_times_ms'])

        printed = False
        if run_id_first != -1:
            printed = True
            assert 'run_ids' in run_data, 'Cannot call populate_summary with a run_id on the top level time_data ' \
                'dictionary.'
            run_id_last = run_id_last if run_id_last != -1 else run_id_first
            run_ids = np.array(run_data['run_ids'])[:len(task_times_ms)]
            run_mask = (run_ids >= run_id_first) & (run_ids <= run_id_last)
            if not np.any(run_mask):
                continue
            task_times_ms = task_times_ms[run_mask]

        summary[task_name] = get_stats(task_times_ms, printed)

        if run_data['subtasks']:
            subtask_time_data = run_data['subtasks']
            subtask_summary = {}
            if run_id_first != -1:
                subtask_run_ids = np.where(run_mask)[0]
                subtask_run_id_first = subtask_run_ids[0]
                subtask_run_id_last = subtask_run_ids[-1]
                populate_summary(subtask_time_data, subtask_summary, subtask_run_id_first, subtask_run_id_last)
            else:
                populate_summary(subtask_time_data, subtask_summary)
            if subtask_summary:
                summary[task_name]['subtasks'] = subtask_summary

    return summary

def save_to_h5(h5_group, time_data):
    for task_name, run_data in time_data.items():
        task_h5_group = None

        if 'task_times_ms' in run_data and run_data['task_times_ms']:
            if task_h5_group is None:
                task_h5_group = h5_group.create_group(task_name)
            task_h5_group.create_dataset(
                'task_times_ms',
                data=np.array(run_data['task_times_ms'], dtype=np.float64),
                compression='gzip')

        if 'run_ids' in run_data and run_data['run_ids']:
            if task_h5_group is None:
                task_h5_group = h5_group.create_group(task_name)
            task_h5_group.create_dataset(
                'run_ids',
                data=np.array(run_data['run_ids'], dtype=np.int32),
                compression='gzip')

        if 'subtasks' in run_data and run_data['subtasks']:
            if task_h5_group is None:
                task_h5_group = h5_group.create_group(task_name)
            subtask_h5_group = task_h5_group.create_group('subtasks')
            save_to_h5(subtask_h5_group, run_data['subtasks'])

def load_from_h5(h5_group):
    time_data = {}
    for task_name in h5_group.keys():
        item = h5_group[task_name]
        if isinstance(item, h5py.Group):
            run_data = {}

            if 'task_times_ms' in item:
                run_data['task_times_ms'] = item['task_times_ms'][:].tolist()
            else:
                run_data['task_times_ms'] = []

            if 'run_ids' in item:
                run_data['run_ids'] = item['run_ids'][:].tolist()

            if 'subtasks' in item:
                run_data['subtasks'] = load_from_h5(item['subtasks'])
            else:
                run_data['subtasks'] = {}

            time_data[task_name] = run_data

    return time_data

def is_main_thread():
    return threading.current_thread() is threading.main_thread()

class NestedTimer:
    # _time_data structure:
    # {
    #   'Task A': {
    #       'task_times_ms': [<list of times for all runs of Task A in ms>],
    #       'subtasks': { # Dictionary of subtasks that appear in at least one run of Task A
    #         'Subtask A': {
    #             'task_time_ms': [<list of times for all runs of Subtask A in ms>],
    #             'run_ids': [<Task A run index for each run of Subtask A>],
    #             'subtasks': { # Dictionary of subsubtasks that appear in at least one run of Subtask A
    #                   `Subsubtask A`: {
    #                       'task_time_ms': [<list of times for all runs of Subsubtask A in ms>],
    #                       'run_ids': [<Subtask A run index for each run of Subsubtask A>],
    #                       'subtasks': { ... }, # Recursively nested
    #                   },
    #                   ...
    #             },
    #         },
    #         ...
    #       },
    #   },
    #   ...
    # }
    _time_data = {}
    _start_times = []
    _run_data_stack = []
    _task_names = []
    disable = False # Should only be directly accessed before any timers are activated.
    force_cuda_sync = False

    _printed_stat_funcs = {
        'mean_ms': lambda x: float(np.mean(x)),
        'median_ms': lambda x: float(np.median(x)),
        'min_ms': lambda x: float(np.min(x)),
        'max_ms': lambda x: float(np.max(x))}
    _saved_stat_funcs = {
        'mean_ms': lambda x: float(np.mean(x)),
        'median_ms': lambda x: float(np.median(x)),
        'min_ms': lambda x: float(np.min(x)),
        'max_ms': lambda x: float(np.max(x))}

    @staticmethod
    def set_custom_stat_funcs(stat_funcs, printed=True):
        if not isinstance(stat_funcs, dict):
            raise TypeError('stat_funcs must be a dictionary.')
        for stat_name, stat_func in stat_funcs.items():
            if not callable(stat_func):
                raise TypeError(f'Value for key \'{stat_name}\' in stat_funcs must be a callable function.')
        if printed:
            NestedTimer._printed_stat_funcs = stat_funcs
        else:
            NestedTimer._saved_stat_funcs = stat_funcs

    def __init__(self, task_name, print_timing_summary=False, enable_timing=True):
        self.task_name = task_name
        self.print_timing_summary = print_timing_summary
        self.enable_timing = enable_timing
        self.global_disable = NestedTimer.disable
        self.started = False

    def __enter__(self):
        if self.enable_timing:
            self.started = NestedTimer._start_task(self.task_name)
        else:
            NestedTimer.disable = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.started:
            NestedTimer._stop_task(self.print_timing_summary)
        NestedTimer.disable = self.global_disable

    @staticmethod
    def timed_function(task_name):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                print_timing_summary = kwargs.pop('print_timing_summary', False)
                enable_timing = kwargs.pop('enable_timing', True)
                with NestedTimer(task_name, print_timing_summary, enable_timing):
                    result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator

    @staticmethod
    def _start_task(task_name):
        if NestedTimer.disable or not is_main_thread():
            return False

        if not NestedTimer._run_data_stack:
            if not (task_name in NestedTimer._time_data):
                NestedTimer._time_data[task_name] = {'task_times_ms': [], 'subtasks': {}}
            run_data = NestedTimer._time_data[task_name]

        else:
            parent_subtasks = NestedTimer._run_data_stack[-1]['subtasks']
            if not (task_name in parent_subtasks):
                parent_subtasks[task_name] = {'task_times_ms': [], 'run_ids': [], 'subtasks': {}}
            run_data = parent_subtasks[task_name]

            parent_task_times_list = NestedTimer._run_data_stack[-1]['task_times_ms']
            run_data['run_ids'].append(len(parent_task_times_list))
        NestedTimer._run_data_stack.append(run_data)
        NestedTimer._task_names.append(task_name)
        NestedTimer._start_times.append(time.perf_counter())
        return True

    @staticmethod
    def _stop_task(print_timing_summary=False):
        if NestedTimer.force_cuda_sync:
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        start_time = NestedTimer._start_times.pop()
        elapsed_ms = (end_time - start_time) * 1000
        run_data = NestedTimer._run_data_stack.pop()
        run_data['task_times_ms'].append(elapsed_ms)
        task_name = NestedTimer._task_names.pop()
        if print_timing_summary:
            print(f'{task_name} took {elapsed_ms} ms.')
            current_run_subtasks = run_data.get('subtasks')
            if current_run_subtasks:
                print('Subtask summary:')
                subtask_summary = {}
                populate_summary(current_run_subtasks, subtask_summary, len(run_data['task_times_ms']) - 1)
                set_percent(subtask_summary, elapsed_ms)
                pprint.pprint(subtask_summary, indent=4, sort_dicts=False)

    @staticmethod
    def clear_timing_data():
        NestedTimer._time_data.clear()
        NestedTimer._start_times.clear()
        NestedTimer._run_data_stack.clear()
        NestedTimer._task_names.clear()

    @staticmethod
    def save_time_data(
        output_folder,
        save_timing_data,
        save_timing_summary,
        timing_data_filename = 'time_data.h5',
        timing_summary_filename = 'time_summary.yaml',
        clear_timing_data=False,
    ):
        if not NestedTimer._time_data:
            return
        os.makedirs(output_folder, exist_ok=True)

        if save_timing_data:
            filename_time_data = os.path.join(output_folder, timing_data_filename)
            if Path(timing_data_filename).suffix == 'h5':
                with h5py.File(filename_time_data, 'w') as file:
                    save_to_h5(file, NestedTimer._time_data)
            else:
                with open(filename_time_data, 'w', encoding='utf8') as file:
                    yaml.dump(NestedTimer._time_data, file, default_flow_style=False, sort_keys=False)

        if save_timing_summary:
            full_summary = {}
            populate_summary(NestedTimer._time_data, full_summary)
            set_percent(full_summary)
            filename_time_summary = os.path.join(output_folder, timing_summary_filename)
            with open(filename_time_summary, 'w', encoding='utf8') as file:
                yaml.dump(full_summary, file, default_flow_style=False, sort_keys=False)

        if clear_timing_data:
            NestedTimer.clear_timing_data()

@contextlib.contextmanager
def nested_timer_paused():
    if not is_main_thread():
        yield
        return

    pause_time = time.perf_counter()
    try:
        yield
    finally:
        resume_time = time.perf_counter()
        pause_duration = resume_time - pause_time
        for i in range(len(NestedTimer._start_times)):
            NestedTimer._start_times[i] += pause_duration
