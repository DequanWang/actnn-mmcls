_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]
actnn = dict(
    default_bit=4,
    auto_prec=False,
)
custom_hooks = [
    dict(
        type="ActnnHook",
        interval=1
    )
]
data = dict(
    samples_per_gpu=64, # 64*4 = 256
    workers_per_gpu=2,
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
                name='resnet50_b64x4_imagenet',
            )
        )
    ]
)
