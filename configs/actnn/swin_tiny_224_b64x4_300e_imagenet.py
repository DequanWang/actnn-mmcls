_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]
actnn = dict(
    default_bit=4,
    auto_prec=False,
)
custom_hooks = [
    dict(
        type="ActNNHook",
        interval=1
    )
]
data = dict(
    samples_per_gpu=64, # 64*4 = 256
    workers_per_gpu=2,
)
optimizer = dict(lr=5e-4 * 64 * 4 / 512)
lr_config = dict(warmup_iters=20 * 5005)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='classification',
                entity='actnn',
                name='swin_tiny_224_b64x4_300e_imagenet',
            )
        )
    ]
)
