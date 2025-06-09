data_root = '/home/data/KITTI-360'
stereo_depth_root = '/home/data/KITTI-360/depth'
seg_label_root = '/home/data/KITTI-360/data_2d_semantics'

camera_used = ['left']

dataset_type = 'KITTI360Dataset'
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

empty_idx = 0

kitti360_class_frequencies = [
        2264087502, 20098728, 104972, 96297, 1149426, 
        4051087, 125103, 105540713, 16292249, 45297267,
        14454132, 110397082, 6766219, 295883213, 50037503,
        1561069, 406330, 30516166, 1950115,
]

# 19 classes with unlabeled
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
    'vegetation', 'terrain', 'pole', 'traffic-sign', 'other-structure', 'other-object'
]
num_class = len(class_names)

# dataset config #
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0
)

data_config={
    'input_size': (384, 1408),
    # 'resize': (-0.06, 0.11),
    # 'rot': (-5.4, 5.4),
    # 'flip': True,
    'resize': (0., 0.),
    'rot': (0.0, 0.0 ),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', data_config=data_config, load_stereo_depth=True,
         is_train=True, color_jitter=(0.4, 0.4, 0.4)),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti360', load_seg=False),
    dict(type='LoadDenseAnnotations', is_train=True),
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=True, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_semantics'], 
            meta_keys=['pc_range', 'occ_size', 'raw_img', 'stereo_depth', 'gt_depths']),
]

trainset_config=dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    data_root=data_root,
    pipeline=train_pipeline,
    split='train',
    camera_used=camera_used,
    pc_range=point_cloud_range,
    test_mode=False,
)

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', data_config=data_config, load_stereo_depth=True,
         is_train=False, color_jitter=None),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti360', load_seg=False),
    dict(type='LoadDenseAnnotations', is_train=False),
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=False, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_semantics'],  
            meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 'stereo_depth'])
]

testset_config=dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    data_root=data_root,
    pipeline=test_pipeline,
    split='test',
    camera_used=camera_used,
    pc_range=point_cloud_range
)

data = dict(
    train=trainset_config,
    val=testset_config,
    test=testset_config
)

train_dataloader_config = dict(
    batch_size=4,
    num_workers=4)

test_dataloader_config = dict(
    batch_size=4,
    num_workers=4)

# model params #
numC_Trans = 128
voxel_channels = [128, 256, 512]
voxel_out_indices = (0, 1, 2)
voxel_out_channels = [128, 128, 128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='CGFormerSegDepth',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(type='Pretrained', prefix='backbone', 
        checkpoint='../../misc/cgformer/ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.5, 1, 2, 4, 4], 
        out_channels=[128, 128, 128, 128, 128]),
    depth_net=dict(
        type='GeometryDepth_Net',
        downsample=8,
        numC_input=640,
        numC_Trans=numC_Trans,
        cam_channels=33,
        grid_config=grid_config,
        loss_depth_type='kld'
    ),
    plugin_head=dict(
        type='plugin_segmentation_head',
        in_channels=numC_Trans,
        out_channel_list=[128, 64, 32],
        num_class=num_class,
    )
)

"""Training params."""
learning_rate=3e-4
training_steps=50000

optimizer = dict(
    type="AdamW",
    lr=learning_rate,
    weight_decay=0.01
)

lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy="cos",
    interval="step",
    frequency=1
) 