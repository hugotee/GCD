default_hooks = dict(
    logger=dict(interval=20, type='LoggerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmdet'
engineering_only = False
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
gcd_experiment_hash = '22a70b577d3d06a1d2bdfb1d6be54e30a0a8a7432bd73f3c36d628980c51833a'
gcd_phase = 'task1'
launcher = 'pytorch'
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor')
model = dict(
    adapter=True,
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=False,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        frozen_stages=-1,
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=False),
    bbox_head=dict(
        contrastive_cfg=dict(bias=None, log_scale=None, max_text_len=256),
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=256,
        sync_cls_avg_factor=True,
        type='GroundingDINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=False,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(
                batch_first=True, dropout=0.0, embed_dims=256, num_heads=8),
            cross_attn_text_cfg=dict(
                batch_first=True, dropout=0.0, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=None,
    encoder=dict(
        fusion_layer_cfg=dict(
            embed_dim=1024,
            init_values=0.0001,
            l_dim=256,
            num_heads=4,
            v_dim=256),
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.0, embed_dims=256, num_levels=4)),
        num_cp=0,
        num_layers=6,
        text_layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.0, embed_dims=256, num_heads=4))),
    expected_classes=[
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
    ],
    initial_checkpoint=
    '/home/data/dq/gcd-ciod-20261104/outputs/voc_four_gpu_v1/pipeline/B1/epoch_12.pth',
    language_model=dict(
        add_pooling_layer=False,
        name='/home/data/dq/cdiod-dgs/weights/bert-base-uncased',
        pad_to_max=False,
        special_tokens_list=[
            '[CLS]',
            '[SEP]',
            '.',
            '?',
        ],
        type='BertModel',
        use_sub_sentence_represent=True),
    lora=dict(alpha=32, dropout=0.0, rank=16),
    neck=dict(
        act_cfg=None,
        bias=True,
        in_channels=[
            192,
            384,
            768,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    train_label_ids=[
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
    ],
    type='SingleLoRAGroundingDINO',
    with_box_refine=True)
model_wrapper_cfg = dict(
    broadcast_buffers=False,
    find_unused_parameters=True,
    type='MMDistributedDataParallel')
randomness = dict(deterministic=False, diff_rank_seed=False, seed=20261103)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/home/data/dq/gcd-ciod-20261104/data/voc15_5/test_full20.json',
        data_prefix=dict(
            img=
            '/home/data/dq/dgs-experiments/gdino_domain_kd_diagnostic_20261101/data/VOCdevkit/VOC2007/JPEGImages/'
        ),
        filter_cfg=None,
        metainfo=dict(
            classes=(
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                1333,
            ), type='FixScaleResize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'text',
                    'custom_entities',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/home/data/dq/gcd-ciod-20261104/data/voc15_5/test_full20.json',
    classwise=True,
    metric='bbox',
    outfile_prefix=
    '/home/data/dq/gcd-ciod-20261104/outputs/voc_four_gpu_v1/pipeline/B1_eval/predictions',
    type='CocoMetric')
visualizer = None
work_dir = '/home/data/dq/gcd-ciod-20261104/outputs/voc_four_gpu_v1/pipeline/B1_eval'
