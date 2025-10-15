# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .realbuilding_components import RealBuildingDatasetComponents
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class QuakeCityDatasetComponents(CustomDataset):
    CLASSES = RealBuildingDatasetComponents.CLASSES
    PALETTE = RealBuildingDatasetComponents.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(QuakeCityDatasetComponents, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)
