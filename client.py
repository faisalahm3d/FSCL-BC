import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from data_utils import ImageMaskLabelDataset, ImageOnlyDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import copy
import os
from losses import SupConLoss
from hyperparameters import *

class Client:
    def __init__(self,
                 client_id,
                 client_model,
                 client_data_dict,
                 transform,
                 local_epoch,
                 con_loss_weight=CONTRASTIVE_LOSS_WEIGHT,
                 input_type='two_channel',
                 work_dir='.',
                 split_seed=TRAIN_TEST_SPLIT_RANDOM_STATE,
                 ):
        super(Client, self).__init__()
        self.client_id = client_id
        self.work_dir = work_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = client_model.to(self.device)
        self.transform = transform
        self.client_data_dict = client_data_dict

        DatasetClass = ImageOnlyDataset if input_type == 'image_only' else ImageMaskLabelDataset
        self.dataset = DatasetClass(
            image_dir=self.client_data_dict['img_dir'],
            mask_dir=self.client_data_dict['mask_dir'],
            label_file=self.client_data_dict['label_file'],
            transform=self.transform)

        indices = list(range(len(self.dataset)))
        train_indices, val_indices = train_test_split(indices,
                                                      test_size=TRAIN_TEST_SPLIT,
                                                      random_state=split_seed)
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=BATCH_SIZE_CLIENT,
                                       shuffle=True, num_workers=NUM_WORKERS)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE_CLIENT,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

        self.local_epoch = local_epoch
        self.criterion = nn.CrossEntropyLoss()
        self.supcon_criterion = SupConLoss(temperature=3)
        self.con_loss_weight = con_loss_weight

        self.best_result = {
            'acc': 0,
            'pre': 0,
            'rec': 0,
            'f1': 0,
            'auc': 0,
            'balanced_acc': 0,
            'mcc': 0,
            'specificity': 0,
            'youden_j': 0
        }

    def recv(self, model_param):
        self.model.load_state_dict(copy.deepcopy(model_param))

    def update(self):
        self.model.train()
        best_loss = float('inf')
        patience = 5
        model_path = os.path.join(self.work_dir, '{}.pth'.format(self.client_id))
        patience_counter = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        for epoch in range(self.local_epoch):
            running_loss = 0.0
            for batch_data in self.train_loader:
                inputs, labels = batch_data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                features, classifications = self.model(inputs)
                features = F.normalize(features, dim=1)
                contrastive_loss = self.supcon_criterion(features.unsqueeze(1), labels)
                classification_loss = self.criterion(classifications, labels)
                loss = self.con_loss_weight * contrastive_loss + classification_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step()
            val_loss = self.evaluate_con_loss()

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), model_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        self.model.load_state_dict(torch.load(model_path))
        result_dict = self.evaluate()
        if result_dict['f1'] > self.best_result['f1']:
            self.best_result = result_dict

    def evaluate_con_loss(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_data in self.val_loader:
                inputs, labels = batch_data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features, outputs = self.model(inputs)
                features = F.normalize(features, dim=1)
                supcon_loss = self.supcon_criterion(features.unsqueeze(1), labels)
                running_loss += supcon_loss.item()
        avg_loss = running_loss / len(self.val_loader)
        self.model.train()
        return avg_loss

    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch_data in self.val_loader:
                inputs, labels = batch_data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, confusion_matrix

        accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
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
            'youden_j': youden_j
        }
