from typing import TYPE_CHECKING

import tyro

from trgs_slam.datasets.base_dataset import BaseDatasetConfig
from trgs_slam.datasets.trnerf_dataset import TRNeRFDatasetConfig

datasets = {
    'trnerf-dataset': TRNeRFDatasetConfig()}

if TYPE_CHECKING:
    DatasetsUnion = BaseDatasetConfig
else:
    DatasetsUnion = tyro.extras.subcommand_type_from_defaults(datasets, prefix_names=False)
