# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .realbuilding_damageStates import RealBuildingDatasetDamageStates
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class QuakeCityDatasetDamageStates(CustomDataset):
    CLASSES = RealBuildingDatasetDamageStates.CLASSES
    PALETTE = RealBuildingDatasetDamageStates.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(QuakeCityDatasetDamageStates, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)
