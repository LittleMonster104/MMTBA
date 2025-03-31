_base_ = ['../../_base_/default_runtime.py']

url = (
    'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/'
    'vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth'
)

model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        type='mmaction.VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6),
        drop_path_rate=0.2,
        use_mean_pooling=False,
        return_feat_map=True),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            background_class=True,
            in_channels=1024,
            num_classes=6,
            multilabel=True,
            dropout_ratio=0.5)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        _scope_='mmaction',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))

dataset_type = 'AVADataset'  # 训练、验证和测试的数据集类型

train_data_root = 'dataset/train/rawframes' #路径需调整
train_anno_root = 'dataset/train/annotations'

val_data_root = 'dataset/val/rawframes'
val_anno_root = 'dataset/val/annotations'
ann_file_train = f'{train_anno_root}/train.csv'  # 训练注释文件的路径
ann_file_val = f'{val_anno_root}/val.csv'  # 验证注释文件的路径

exclude_file_train = f'{train_anno_root}/train_excluded_timestamps.csv'  # 训练排除注释文件的路径
exclude_file_val = f'{val_anno_root}/val_excluded_timestamps.csv'  # 验证排除注释文件的路径

label_file = f'{train_anno_root}/action_list.pbtxt'  # 标签文件的路径

proposal_file_train = f'{train_anno_root}/dense_proposals_train.pkl'  # 训练示例的人体检测 proposals 文件的路径
proposal_file_val = f'{val_anno_root}/dense_proposals_val.pkl'  # 验证示例的人体检测 proposals 文件的路径

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=16, frame_interval=4),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=16, frame_interval=4, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=train_data_root)))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=val_data_root),
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=exclude_file_val)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=15,
        eta_min=0,
        by_epoch=True,
        begin=5,
        end=20,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=2.5e-4, weight_decay=0.05),
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.8,
        'decay_type': 'layer_wise',
        'num_layers': 24
    },
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
