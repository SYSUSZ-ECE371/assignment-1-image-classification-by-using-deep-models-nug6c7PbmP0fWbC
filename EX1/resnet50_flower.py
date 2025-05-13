_base_ = [
    'C:\\Reina_desktop\\program_and_contest\\DL_homework\\assignment_1\\mmclassification\\configs\\_base_\\models\\resnet50.py',
    'C:\\Reina_desktop\\program_and_contest\\DL_homework\\assignment_1\\mmclassification\\configs\\_base_\\datasets\\imagenet_bs16_eva_196.py',
    'C:\\Reina_desktop\\program_and_contest\\DL_homework\\assignment_1\\mmclassification\\configs\\_base_\\schedules\\cifar10_bs128.py',
    'C:\\Reina_desktop\\program_and_contest\\DL_homework\\assignment_1\\mmclassification\\configs\\_base_\\default_runtime.py',
]

model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=5),
)

# data pipeline settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=224,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

# dataloader settings
train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    # Training dataset configurations
    dataset=dict(
        type='ImageNet',
        data_root='C:\\Reina_desktop\\program_and_contest\\DL_homework\\assignment_1\\part_1\\flower_dataset',
        ann_file='train.txt',
        data_prefix='train/',
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=1,
    # Validation dataset configurations
    dataset=dict(
        type='ImageNet',
        data_root='C:\\Reina_desktop\\program_and_contest\\DL_homework\\assignment_1\\part_1\\flower_dataset',
        ann_file='val.txt',
        data_prefix='val\\',
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader