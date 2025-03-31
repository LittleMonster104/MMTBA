_base_ = ['slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py']

model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(with_global=True, temporal_pool_mode='max'),
        bbox_head=dict(in_channels=4096, mlp_head=True, focal_gamma=1.0)))

dataset_type = 'AVADataset'
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

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=16, frame_interval=4),
    dict(type='RawFrameDecode', **file_client_args),
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
    dict(type='RawFrameDecode', **file_client_args),
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
    type='EpochBasedTrainLoop', max_epochs=10, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=0,
        by_epoch=True,
        begin=2,
        end=10,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.00001),
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
