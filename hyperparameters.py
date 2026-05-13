#!/usr/bin/env python3
"""
Central hyperparameter file for FSCL-BC pipeline.
Edit dataset paths and training parameters here.
"""

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 0
NUMPY_SEED = 0
CUDNN_DETERMINISTIC = True

# ============================================================================
# DATASET PATHS (Breast USG)
# Update these paths to match your server/machine before running.
# ============================================================================
CLIENT_IMG_DIRS = [
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/img/Toshiba Aplio 300 @12-14 MHz',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUS-UCLM/img_gray',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/img/GE Logiq 5 @10-12MHz',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/img/GE Logiq 7 @10-14MHz',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/img/U-Systems @10-14MHz',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSI/img_gray',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/QAMEBI/img_gray',
    '/home/trail/FaisalM/FedMedImaging/multi_site_busi/Image/BUSUSG'
]

CLIENT_MASK_DIRS = [
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/seg/Toshiba Aplio 300 @12-14 MHz',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUS-UCLM/seg_gray',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/seg/GE Logiq 5 @10-12MHz',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/seg/GE Logiq 7 @10-14MHz',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/seg/U-Systems @10-14MHz',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSI/seg_gray',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/QAMEBI/seg_gray',
    '/home/trail/FaisalM/FedMedImaging/multi_site_busi/Label/BUSUSG'
]

CLIENT_LABEL_FILES = [
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/ToshibaAplio300.txt',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUS-UCLM/UCLM-two-class.txt',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/GELogiq5.txt',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/GELogiq7.txt',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSBRA/U-Systems.txt',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/BUSI/BUSI.txt',
    '/home/trail/FL_DataSets/multi_center_breast_usg_datasets/QAMEBI/QAMEBI.txt',
    '/home/trail/FaisalM/FedMedImaging/multi_site_busi/Annotation/BUSUSG.txt'
]

# ============================================================================
# IMAGE PROCESSING
# ============================================================================
IMAGE_SIZE = (128, 128)
NORMALIZE_MEAN = [0.5]
NORMALIZE_STD = [0.5]

AUGMENTATION = {
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True
}

# ============================================================================
# DATALOADER PARAMETERS
# ============================================================================
BATCH_SIZE_CLIENT = 64
BATCH_SIZE_SERVER = 32
BATCH_SIZE_GLOBAL = 32
NUM_WORKERS = 4
TRAIN_TEST_SPLIT = 0.2
TRAIN_TEST_SPLIT_RANDOM_STATE = 42

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
LOCAL_EPOCH = 10
LEARNING_RATE = 0.001
PATIENCE = 5
CONTRASTIVE_LOSS_WEIGHT = 0.1
CRITERION = 'CrossEntropyLoss'

# ============================================================================
# FEDERATED LEARNING PARAMETERS
# ============================================================================
FL_METHOD = 'FSCL-BC'
COMMUNICATION_ROUNDS = 10
CLIENT_FRACTION = 1.0

# ============================================================================
# MODEL ARCHITECTURE
# Options: 'resnet18', 'efficientnet', 'densenet'
# ============================================================================
CLIENT_MODEL = 'resnet18'
SERVER_MODEL = 'resnet18'

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
TASK_NAME = 'fscl_bc'
SAVE_DIR = 'resources'
SAVE_BEST_MODEL = True
MODEL_SAVE_FORMAT = 'pth'

# ============================================================================
# DEVICE
# ============================================================================
DEVICE = 'cuda'
