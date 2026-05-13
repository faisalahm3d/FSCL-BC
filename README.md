# FSCL-BC

Federated Supervised Contrastive Learning for binary classification (Benign vs. Malignant) of breast ultrasound images.

One domain is held out as the server/test set; all remaining domains participate as FL clients using FSCL (CE loss + SupConLoss).

## Requirements

```bash
pip install torch torchvision scikit-learn pillow numpy
```

## Setup

### 1. Update dataset paths

Open `hyperparameters.py` and update the three lists to point to your data:

```python
CLIENT_IMG_DIRS   = [ ... ]   # paths to image directories (one per domain)
CLIENT_MASK_DIRS  = [ ... ]   # paths to segmentation mask directories
CLIENT_LABEL_FILES = [ ... ]  # paths to label .txt files (comma-separated: image_name,label)
```

Label file format (one line per image):
```
image001.png,0
image002.png,1
```
where `0` = Benign, `1` = Malignant.

The 8 domains currently configured are:

| Index | Dataset |
|---|---|
| 0 | BUSBRA — Toshiba Aplio 300 @12-14 MHz |
| 1 | BUS-UCLM |
| 2 | BUSBRA — GE Logiq 5 @10-12 MHz |
| 3 | BUSBRA — GE Logiq 7 @10-14 MHz |
| 4 | BUSBRA — U-Systems @10-14 MHz |
| 5 | BUSI |
| 6 | QAMEBI |
| 7 | BUSUSG |

### 2. Adjust hyperparameters (optional)

All training settings are in `hyperparameters.py`:

| Parameter | Default | Description |
|---|---|---|
| `CONTRASTIVE_LOSS_WEIGHT` | `0.1` | Weight of SupConLoss (alpha) |
| `COMMUNICATION_ROUNDS` | `10` | Number of FL communication rounds |
| `LOCAL_EPOCH` | `10` | Local training epochs per client per round |
| `LEARNING_RATE` | `0.001` | Adam optimizer learning rate |
| `BATCH_SIZE_CLIENT` | `64` | Client batch size |
| `CLIENT_FRACTION` | `1.0` | Fraction of clients selected per round |
| `RANDOM_SEED` | `0` | Torch random seed |

The SupConLoss temperature is fixed at `3` and the StepLR scheduler decays the learning rate by 0.1 every 5 local epochs. Early stopping uses patience of 5 on the contrastive validation loss.

## Running

```bash
python run.py --fold N
```

`N` is the 0-based index of the domain to hold out as the server/test set (0 to 7). All other domains train as FL clients.

**Examples:**
```bash
python run.py --fold 0   # domain 0 held out; domains 1-7 are clients
python run.py --fold 3   # domain 3 held out; domains 0-2, 4-7 are clients
```

## Output

Results are saved to `resources/result_fold_N.pkl` — a dictionary with the best metrics across all communication rounds (selected by F1):

| Key | Description |
|---|---|
| `f1` | F1 score |
| `auc` | ROC-AUC |
| `acc` | Accuracy (%) |
| `balanced_acc` | Balanced accuracy |
| `rec` | Sensitivity / Recall |
| `specificity` | Specificity |
| `mcc` | Matthews Correlation Coefficient |
| `youden_j` | Youden's J statistic |
| `tn`, `fp`, `fn`, `tp` | Confusion matrix counts |
| `fpr`, `tpr`, `thresholds` | ROC curve arrays |

Client checkpoints (`client_0.pth` ... `client_N.pth`) are written to the working directory during training and can be deleted afterwards.

## File Structure

```
FSCL-BC/
  run.py              # Entry point
  hyperparameters.py  # All configuration
  config.py           # Experiment orchestration
  server.py           # FL server (FedAvg aggregation + global evaluation)
  client.py           # FL client (CE + SupConLoss, early stopping)
  model.py            # ResNet18 backbone (pseudo 3-channel)
  data_utils.py       # Dataset class (image + mask → 2-channel tensor)
  losses.py           # SupConLoss implementation
  resources/          # Output directory (created automatically)
```
