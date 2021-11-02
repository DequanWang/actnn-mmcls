_base_ = [
    '../_base_/models/resnet18.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs2048.py',
    '../_base_/default_runtime.py'
]
actnn = True
data = dict(
    samples_per_gpu=512, # 512*4 = 2048
    workers_per_gpu=8,
)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='classification',
                entity='actnn',
                name='resnet18_b512x4_imagenet',
            )
        )
    ]
)
