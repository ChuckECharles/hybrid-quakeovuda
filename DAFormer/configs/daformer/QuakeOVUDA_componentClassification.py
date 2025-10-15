# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
#
# Modifications © 2025 Charles Abdo
# Changes: Adapted for Quake-OVUDA integration.

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture (Component Classification Config)
    '../_base_/models/daformer_sepaspp_mitb5_componentClassification.py',
    # QuakeCity->RealBuilding Data Loading
    '../_base_/datasets/uda_quakecity_to_realbuilding_components_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    # Set these to zero because feature distance doesn't apply to our testing
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=[],
    imnet_feature_dist_scale_min_ratio=0,
    # Pseudo-Label Crop
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    # Use SED produced pseudolabels during training
    use_sed_labels = True)
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=35000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=35000, max_keep_ckpts=1)

evaluation = dict(interval=5000, metric='mIoU')
# Meta Information for Result Analysis
name = 'QuakeOVUDA_componentClassification'
exp = 'basic'
name_dataset = 'quake2realbuilding'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
