import torch
import numpy as np
from server import Server
from torchvision import transforms
from model import *
from hyperparameters import *
import warnings
import time

warnings.filterwarnings("ignore")

torch.manual_seed(RANDOM_SEED)
np.random.seed(NUMPY_SEED)
torch.backends.cudnn.deterministic = CUDNN_DETERMINISTIC


class Config:
    def __init__(self,
                 task_name,
                 client_data_dicts,
                 global_data_dict,
                 method,
                 con_loss_weight=CONTRASTIVE_LOSS_WEIGHT,
                 input_type='two_channel',
                 work_dir='.',
                 train_test_split_seed=TRAIN_TEST_SPLIT_RANDOM_STATE):
        super(Config, self).__init__()
        self.task_name = task_name
        self.client_data_dicts = client_data_dicts
        self.global_data_dict = global_data_dict
        self.method = method
        self.con_loss_weight = con_loss_weight
        self.input_type = input_type
        self.work_dir = work_dir
        self.train_test_split_seed = train_test_split_seed

    def exp(self, fold=0):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        client_num = len(self.client_data_dicts)

        model = get_3channel_resnet18()

        params = {
            'task':               self.task_name,
            'client_model':       model,
            'server_model':       model,
            'client_num':         client_num,
            'client_data_dicts':  self.client_data_dicts,
            'server_data_dict':   self.global_data_dict,
            'transform':          transform,
            'local_epoch':        LOCAL_EPOCH,
            'C':                  CLIENT_FRACTION,
            'communication_round': COMMUNICATION_ROUNDS,
            'method':             self.method,
            'con_loss_weight':    self.con_loss_weight,
            'input_type':         self.input_type,
            'work_dir':           self.work_dir,
            'split_seed':         self.train_test_split_seed,
        }

        fl_entity = Server(params)

        best_results = {
            'acc': 0, 'pre': 0, 'rec': 0, 'f1': 0, 'auc': 0,
            'balanced_acc': 0, 'mcc': 0, 'specificity': 0, 'youden_j': 0,
            'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0,
            'fpr': 0, 'tpr': 0, 'thresholds': 0,
        }

        best_metric = 0
        start_time = time.time()
        best_round = 0

        for t in range(params['communication_round']):
            result_dict = fl_entity.global_update()
            if result_dict['f1'] > best_metric:
                best_results.update(result_dict)
                best_round = t
                best_metric = result_dict['f1']

        print(f'Best round: {best_round}  F1={best_results["f1"]:.4f}  '
              f'AUC={best_results["auc"]:.4f}  '
              f'Time: {time.time() - start_time:.1f}s')

        return best_results
