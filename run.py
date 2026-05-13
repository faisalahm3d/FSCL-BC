#!/usr/bin/env python3
"""
Single-fold FSCL training.

Usage:
    python run.py --fold N

    N : 0-7  Domain index held out as the server / test set.
              All other domains participate as FL clients.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from hyperparameters import (
    RANDOM_SEED, NUMPY_SEED, CUDNN_DETERMINISTIC,
    CLIENT_IMG_DIRS, CLIENT_MASK_DIRS, CLIENT_LABEL_FILES,
    TASK_NAME, FL_METHOD,
)
from config import Config


def main():
    parser = argparse.ArgumentParser(description='FSCL single-fold experiment')
    parser.add_argument('--fold', type=int, required=True,
                        help='Domain index to hold out as server/test set (0-7)')
    args = parser.parse_args()

    n_domains = len(CLIENT_IMG_DIRS)
    if args.fold < 0 or args.fold >= n_domains:
        print(f'ERROR: --fold must be 0-{n_domains - 1}')
        sys.exit(1)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(NUMPY_SEED)
    torch.backends.cudnn.deterministic = CUDNN_DETERMINISTIC

    fold_idx = args.fold

    global_data_dict = {
        'img_dir':    CLIENT_IMG_DIRS[fold_idx],
        'mask_dir':   CLIENT_MASK_DIRS[fold_idx],
        'label_file': CLIENT_LABEL_FILES[fold_idx],
    }
    client_data_dicts = [
        {
            'img_dir':    CLIENT_IMG_DIRS[j],
            'mask_dir':   CLIENT_MASK_DIRS[j],
            'label_file': CLIENT_LABEL_FILES[j],
        }
        for j in range(n_domains) if j != fold_idx
    ]

    print('=' * 60)
    print(f'FSCL  --  fold {fold_idx}  (held-out domain = {fold_idx})')
    print(f'  Clients : {len(client_data_dicts)}')
    print(f'  Server  : domain {fold_idx}')
    print('=' * 60)

    config = Config(
        task_name=TASK_NAME,
        client_data_dicts=client_data_dicts,
        global_data_dict=global_data_dict,
        method=FL_METHOD,
    )
    result = config.exp(fold=fold_idx)

    out_dir = Path('resources')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'result_fold_{fold_idx}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)

    print(f'\nResult saved to {out_path}')
    print(f'  F1={result["f1"]:.4f}  AUC={result["auc"]:.4f}  '
          f'Balanced Acc={result["balanced_acc"]:.4f}')


if __name__ == '__main__':
    main()
