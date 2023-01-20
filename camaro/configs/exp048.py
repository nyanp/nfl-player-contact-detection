from copy import deepcopy
import albumentations as A
from camaro.datasets import video_transforms
from .base import cfg

FPS = 59.94
HEIGHT, WIDTH = 704, 1280
ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 720, 1280
DURATION = 1


cfg = deepcopy(cfg)
cfg.project = 'kaggle-nfl2022'
cfg.exp_name = 'exp048_both_ext_blur_dynamic_normalize_coords_fix_frame_noise'
cfg.output_dir = f'output/{cfg.exp_name}'
cfg.debug = False

cfg.train.df_path = '../input/folds.csv'
cfg.train.label_df_path = '../input/train_labels_with_folds.csv'
cfg.train.helmet_df_path = '../input/train_helmets_with_folds.csv'
cfg.train.tracking_df_path = '../input/train_tracking_with_folds.csv'
cfg.train.image_dir = '../input/train_frames'
cfg.train.duration = DURATION
cfg.train.batch_size = 4
cfg.train.num_workers = 4 if not cfg.debug else 0
cfg.train.down_ratio = 8
cfg.train.roi_size = 'dynamic'
cfg.train.image_size = (HEIGHT, WIDTH)
cfg.train.original_image_size = (ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
cfg.train.transforms = video_transforms.Compose([
    video_transforms.Resize((HEIGHT, WIDTH)),
])
cfg.train.image_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=1.0, rotate_limit=5),
    A.RandomBrightnessContrast(p=1.0),
    A.AdvancedBlur(),
], bbox_params=A.BboxParams(format='pascal_voc'))
cfg.train.enable_frame_noise = True
cfg.train.normalize_coords = True

cfg.valid.df_path = '../input/folds.csv'
cfg.valid.label_df_path = '../input/train_labels_with_folds.csv'
cfg.valid.helmet_df_path = '../input/train_helmets_with_folds.csv'
cfg.valid.tracking_df_path = '../input/train_tracking_with_folds.csv'

cfg.valid.unique_ids_dict_name = 'unique_ids_dict'
cfg.valid.inter_contact_dict_name = 'inter_contact_dict'
cfg.valid.ground_contact_dict_name = 'ground_contact_dict'
cfg.valid.helmet_dict_name = 'helmet_dict'
cfg.valid.track_dict_name = 'track_dict'
cfg.valid.track_dict_by_step_name = 'track_dict_by_step'

cfg.valid.image_dir = '../input/train_frames'
cfg.valid.duration = DURATION
cfg.valid.batch_size = 4
cfg.valid.num_workers = 4 if not cfg.debug else 0
cfg.valid.down_ratio = 8
cfg.valid.roi_size = 'dynamic'
cfg.valid.image_size = (HEIGHT, WIDTH)
cfg.valid.original_image_size = (ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
cfg.valid.transforms = video_transforms.Compose([
    video_transforms.Resize((HEIGHT, WIDTH)),
])
cfg.valid.image_transforms = None
cfg.valid.normalize_coords = True

cfg.model.inter_weight = 0.5
cfg.model.ground_weight = 0.5
cfg.model.pretrained_path = '../input/cut_yolox_m_backbone.pth'
cfg.model.duration = DURATION
cfg.model.down_ratio = 8

# others
cfg.seed = 42
cfg.device = 'cuda'
cfg.lr = 1.0e-4
cfg.wd = 1.0e-4
cfg.min_lr = 5.0e-5
cfg.warmup_lr = 1.0e-5
cfg.warmup_epochs = 3
cfg.warmup = 1
cfg.epochs = 15
cfg.eval_intervals = 1
cfg.mixed_precision = True
cfg.ema_start_epoch = 1
