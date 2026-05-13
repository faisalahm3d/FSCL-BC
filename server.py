import copy
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from data_utils import ImageMaskLabelDataset, ImageOnlyDataset
from hyperparameters import *
from client import Client
import torch


class Server:
    def __init__(self, params):
        super(Server, self).__init__()
        self.transform = params['transform']
        self.global_data_dict = params['server_data_dict']
        input_type = params.get('input_type', 'two_channel')
        DatasetClass = ImageOnlyDataset if input_type == 'image_only' else ImageMaskLabelDataset
        self.global_dataset = DatasetClass(
            image_dir=self.global_data_dict['img_dir'],
            mask_dir=self.global_data_dict['mask_dir'],
            label_file=self.global_data_dict['label_file'],
            transform=self.transform)

        self.global_test_loader = DataLoader(self.global_dataset,
                                             batch_size=BATCH_SIZE_SERVER,
                                             shuffle=True,
                                             num_workers=NUM_WORKERS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = params['server_model'].to(self.device)
        self.client_num = params['client_num']
        self.C = params['C']
        self.T = params['communication_round']
        con_loss_weight = params.get('con_loss_weight', CONTRASTIVE_LOSS_WEIGHT)
        work_dir = params.get('work_dir', '.')
        split_seed = params.get('split_seed', TRAIN_TEST_SPLIT_RANDOM_STATE)

        self.clients = [
            Client(
                'client_{}'.format(i),
                copy.deepcopy(params['client_model']),
                params['client_data_dicts'][i],
                params['transform'],
                params['local_epoch'],
                con_loss_weight=con_loss_weight,
                input_type=input_type,
                work_dir=work_dir,
                split_seed=split_seed,
            )
            for i in range(self.client_num)
        ]

        self.weight = np.array([1.0 for _ in self.clients])
        self.broadcast(self.global_model.state_dict())

    def aggregated(self, idxs_users):
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                new_par[name] += par[name] * (w / self.C)
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        return self.global_model.state_dict().copy()

    def broadcast(self, new_par):
        for client in self.clients:
            client.recv(new_par.copy())

    def global_update(self):
        idxs_users = np.sort(
            np.random.choice(range(len(self.clients)),
                             int(self.C * len(self.clients)),
                             replace=False))
        for idx in idxs_users:
            self.clients[idx].update()
        self.broadcast(self.aggregated(idxs_users))
        result_dict = self.test_acc()
        torch.cuda.empty_cache()
        return result_dict

    def test_acc(self):
        self.global_model.eval()
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch_data in self.global_test_loader:
                inputs, labels = batch_data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, outputs = self.global_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, confusion_matrix
        accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)
        roc_auc = roc_auc_score(all_labels, all_probabilities)
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
        mcc = matthews_corrcoef(all_labels, all_predictions)

        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
        specificity = tn / (tn + fp)
        youden_j = recall + specificity - 1

        return {
            'acc': accuracy,
            'pre': precision,
            'rec': recall,
            'f1': f1,
            'auc': roc_auc,
            'balanced_acc': balanced_acc,
            'mcc': mcc,
            'specificity': specificity,
            'youden_j': youden_j,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds,
        }
