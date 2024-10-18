import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr
import logging

from model.util import Normalizer, seed_everything
from model.database_util import (
    generate_column_min_max,
    get_job_table_sample_direct,
    generate_histograms_entire_db,
    filterDict2Hist,
    collator,
    save_histograms,
    load_entire_histograms
)
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Define schema
imdb_schema = {
    'title': ['id', 'kind_id', 'production_year'],
    'movie_companies': ['id', 'company_id', 'movie_id', 'company_type_id'],
    'cast_info': ['id', 'movie_id', 'person_id', 'role_id'],
    'movie_info_idx': ['id', 'movie_id', 'info_type_id'],
    'movie_info': ['id', 'movie_id', 'info_type_id'],
    'movie_keyword': ['id', 'movie_id', 'keyword_id']
}

# Define table aliases
t2alias = {
    'title': 't',
    'movie_companies': 'mc',
    'cast_info': 'ci',
    'movie_info_idx': 'mi_idx',
    'movie_info': 'mi',
    'movie_keyword': 'mk'
}

alias2t = {v: k for k, v in t2alias.items()}

# Initialize Encoding
col2idx = {
    't.id': 0, 't.kind_id': 1, 't.production_year': 2,
    'mc.id': 3, 'mc.company_id': 4, 'mc.movie_id': 5, 'mc.company_type_id': 6,
    'ci.id': 7, 'ci.movie_id': 8, 'ci.person_id': 9, 'ci.role_id': 10,
    'mi.id': 11, 'mi.movie_id': 12, 'mi.info_type_id': 13,
    'mi_idx.id': 14, 'mi_idx.movie_id': 15, 'mi_idx.info_type_id': 16,
    'mk.id': 17, 'mk.movie_id': 18, 'mk.keyword_id': 19, 'NA': 20
}

# Define Args
class Args:
    bs = 128
    lr = 0.001
    epochs = 100
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    newpath = './results/full/cost/'
    to_predict = 'cost'
args = Args()

# Ensure results directory exists
os.makedirs(args.newpath, exist_ok=True)

# Database connection parameters
DB_PARAMS = {
    'database': "imdb",
    'user': "wuy",
    'host': "127.0.0.1",
    'password': "wuy",
    'port': "5432"
}

# Load column_min_max_vals from CSV
def load_column_min_max(file_path):
    """
    Loads column min and max values from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: Dictionary mapping column names to (min, max).
    """
    df = pd.read_csv(file_path)
    column_min_max_vals = {}
    for _, row in df.iterrows():
        column_min_max_vals[row['name']] = (row['min'], row['max'])
    return column_min_max_vals

column_min_max_file = './data/imdb/column_min_max_vals.csv'
if not os.path.exists(column_min_max_file):
    logging.info(f"Generating column min-max values and saving to '{column_min_max_file}'.")
    generate_column_min_max(DB_PARAMS, imdb_schema, output_file=column_min_max_file, t2alias=t2alias)

column_min_max_vals = load_column_min_max(column_min_max_file)
logging.info(f"Loaded column min-max values from '{column_min_max_file}'.")

# Initialize Normalizers
cost_norm = Normalizer(-3.61192, 12.290855)  # Example values, adjust as needed
card_norm = Normalizer(1, 100)  # Example values, adjust as needed

# Perform direct sampling
sample_dir = './data/imdb/sampled_data/'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Load the synthetic.csv with appropriate column names
column_names = ['id', 'tables_joins', 'predicate', 'cardinality']
query_file_path = './data/imdb/workloads/synthetic.csv'
query_file = pd.read_csv(query_file_path, sep='#', header=None, names=column_names)

# Verify the DataFrame columns
# print(query_file.columns)  # Should output: ['id', 'tables_joins', 'predicate', 'cardinality']

# Continue with the rest of your processing
sampled_data = get_job_table_sample_direct(
    DB_PARAMS,
    imdb_schema,
    query_file,
    alias2t,
    num_samples=1000,
    sample_dir=sample_dir
)

logging.info("Completed sampling for all queries.")

# Generate histograms based on entire tables
hist_dir = './data/imdb/histograms/'
histogram_file_path = './data/imdb/histogram_entire.csv'

if not os.path.exists(histogram_file_path):
    hist_file_df = generate_histograms_entire_db(
        DB_PARAMS,
        imdb_schema,
        hist_dir=hist_dir,
        bin_number=50,
        t2alias=t2alias
    )
    save_histograms(hist_file_df, save_path=histogram_file_path)
else:
    hist_file_df = load_entire_histograms(load_path=histogram_file_path)

encoding = Encoding(column_min_max_vals=column_min_max_vals, col2idx=col2idx)
logging.info("Initialized Encoding object.")

# Seed for reproducibility
seed_everything()

# Load training data (adjust the range as needed)
imdb_path = './data/imdb/'
train_parts = [f"{imdb_path}plan_and_cost/train_plan_part{i}.csv" for i in range(2)]  # Example: parts 0 and 1
dfs = [pd.read_csv(file) for file in train_parts]
full_train_df = pd.concat(dfs, ignore_index=True)

# Load validation data (adjust the range as needed)
val_parts = [f"{imdb_path}plan_and_cost/train_plan_part{i}.csv" for i in range(18, 20)]
val_dfs = [pd.read_csv(file) for file in val_parts]
val_df = pd.concat(val_dfs, ignore_index=True)

logging.info(f"Loaded training data with {len(full_train_df)} records.")
logging.info(f"Loaded validation data with {len(val_df)} records.")

# Initialize PlanTreeDataset
train_ds = PlanTreeDataset(
    json_df=full_train_df,
    train=None,
    encoding=encoding,
    hist_file=hist_file_df,
    card_norm=card_norm,
    cost_norm=cost_norm,
    to_predict=args.to_predict,
    table_sample=sampled_data  # This should be a list indexed by query_id
)

val_ds = PlanTreeDataset(
    json_df=val_df,
    train=None,
    encoding=encoding,
    hist_file=hist_file_df,
    card_norm=card_norm,
    cost_norm=cost_norm,
    to_predict=args.to_predict,
    table_sample=sampled_data  # Ensure consistency
)

logging.info("Initialized training and validation datasets with sampled data.")

# Initialize the model
model = QueryFormer(
    emb_size=args.embed_size,
    ffn_dim=args.ffn_dim,
    head_size=args.head_size,
    dropout=args.dropout,
    n_layers=args.n_layers,
    use_sample=True,
    use_hist=True,
    pred_hid=args.pred_hid
).to(args.device)

logging.info("Initialized QueryFormer model.")

# Define loss function
crit = nn.MSELoss()

# Train the model
model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args)
logging.info(f"Training completed. Best model saved at: {best_path}")

# Define methods dictionary for evaluation
methods = {
    'get_sample': lambda workload_file: get_job_table_sample_direct(
        DB_PARAMS, imdb_schema, workload_file, alias2t, num_samples=1000, sample_dir=sample_dir, t2alias=t2alias
    ),
    'encoding': encoding,
    'cost_norm': cost_norm,
    'hist_file': hist_file_df,
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
