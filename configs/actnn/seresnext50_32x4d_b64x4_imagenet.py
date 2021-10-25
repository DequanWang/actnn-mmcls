_base_ = [
    '../_base_/models/seresnext50_32x4d.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]
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
                name='seresnext50_32x4d_b64x4_imagenet',
            )
        )
    ]
)
