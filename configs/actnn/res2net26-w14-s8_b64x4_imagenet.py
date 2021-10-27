_base_ = [
    '../_base_/models/res2net50-w14-s8.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]
model = dict(backbone=dict(depth=26))
actnn = True
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
                name='res2net26-w14-s8_b64x4_imagenet',
            )
        )
    ]
)
