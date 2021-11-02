_base_ = [
    '../_base_/models/repvgg-A0_in1k.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs2048_coslr.py',
    '../_base_/default_runtime.py'
]
runner = dict(max_epochs=120)

actnn = True
data = dict(
    samples_per_gpu=512, # 512*4 = 2048
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
                name='repvggA0_b512x4_imagenet',
            )
        )
    ]
)
evaluation = dict(interval=10)
