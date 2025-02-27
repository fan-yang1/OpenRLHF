""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    data = load_dataset('K-and-K/knights-and-knaves', 'train')


    all_splits = list(data.values())
    combined_dataset = concatenate_datasets(all_splits)

    combined_dataset.save_to_disk("./data/kk_train")

