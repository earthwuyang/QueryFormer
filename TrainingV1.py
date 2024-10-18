##################
# TrainingV1.py  #
##################
import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr
import logging

from model.util import Normalizer, seed_everything
from model.database_util import get_job_table_sample_direct, generate_histograms_from_sample, collator
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train

seed_everything()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')



data_path = './data/imdb/'


class Args:
    # bs = 1024
    # SQ: smaller batch size
    bs = 128
    lr = 0.001
    # epochs = 200
    epochs = 100
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cuda:0'
    newpath = './results/full/cost/'
    to_predict = 'cost'
args = Args()

if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)

# Database connection parameters
DB_PARAMS = {
    'database': "imdb",
    'user': "wuy",
    'host': "127.0.0.1",
    'password': "wuy",
    'port': "5432"
}

# Define schema and aliases
imdb_schema = {
    'title': ['id', 'kind_id', 'production_year'],
    'movie_companies': ['id', 'company_id', 'movie_id', 'company_type_id'],
    'cast_info': ['id', 'movie_id', 'person_id', 'role_id'],
    'movie_info_idx': ['id', 'movie_id', 'info_type_id'],
    'movie_info': ['id', 'movie_id', 'info_type_id'],
    'movie_keyword': ['id', 'movie_id', 'keyword_id']
}

t2alias = {
    'title': 't',
    'movie_companies': 'mc',
    'cast_info': 'ci',
    'movie_info_idx': 'mi_idx',
    'movie_info': 'mi',
    'movie_keyword': 'mk'
}

alias2t = {v: k for k, v in t2alias.items()}



if os.path.exists(os.path.join(data_path, 'histogram_direct.csv')):
    hist_file = pd.read_csv(os.path.join(data_path, 'histogram_direct.csv'))
    logging.info("Loaded histograms from 'histogram_direct.csv'.")
else:
    # Perform direct sampling
    sampled_data = get_job_table_sample_direct(DB_PARAMS, imdb_schema, num_samples=1000)
    # Generate histograms from sampled data
    hist_file = generate_histograms_from_sample(sampled_data, bin_number=50)

    # Save histograms to CSV for future use (optional)
    hist_file.to_csv(os.path.join(data_path, 'histogram_direct.csv'), index=False)
    logging.info("Saved generated histograms to 'histogram_direct.csv'.")




cost_norm = Normalizer(-3.61192, 12.290855)
card_norm = Normalizer(1,100)


# encoding_ckpt = torch.load('checkpoints/encoding.pt')
# encoding = encoding_ckpt['encoding']
# checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu')

# Initialize Encoding
encoding = Encoding(column_min_max_vals=column_min_max_vals, col2idx=col2idx)


# Load training data (adjust the range as needed)
train_parts = [f"{data_path}plan_and_cost/train_plan_part{i}.csv" for i in range(2)]  # Example: parts 0 and 1
dfs = [pd.read_csv(file) for file in train_parts]
full_train_df = pd.concat(dfs, ignore_index=True)

# Load validation data (adjust the range as needed)
val_parts = [f"{data_path}plan_and_cost/train_plan_part{i}.csv" for i in range(18, 20)]
val_dfs = [pd.read_csv(file) for file in val_parts]
val_df = pd.concat(val_dfs, ignore_index=True)

logging.info(f"Loaded training data with {len(full_train_df)} records.")
logging.info(f"Loaded validation data with {len(val_df)} records.")



train_ds = PlanTreeDataset(full_train_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, sampled_data)
val_ds = PlanTreeDataset(val_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, sampled_data)

model = QueryFormer(emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, \
                 dropout = args.dropout, n_layers = args.n_layers, \
                 use_sample = True, use_hist = True, \
                 pred_hid = args.pred_hid
                )

_ = model.to(args.device)

to_predict = 'cost'


crit = nn.MSELoss()
model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args)


methods = {
    # 'get_sample' : get_job_table_sample_direct,
    'get_sample' : lambda workload_file: get_job_table_sample_direct(DB_PARAMS, imdb_schema, num_samples=1000),
    'encoding': encoding,
    'cost_norm': cost_norm,
    'hist_file': hist_file,
    'model': model,
    'device': args.device,
    'bs': 512,
}


# Evaluate on 'job-light' workload
job_light_scores, job_light_corr = eval_workload('job-light', methods)
logging.info(f"Job-Light Workload Evaluation: {job_light_scores}, Correlation: {job_light_corr}")

# Evaluate on 'synthetic' workload
synthetic_scores, synthetic_corr = eval_workload('synthetic', methods)
logging.info(f"Synthetic Workload Evaluation: {synthetic_scores}, Correlation: {synthetic_corr}")