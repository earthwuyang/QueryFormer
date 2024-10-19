import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr
import logging
import json

import ast  # Needed for literal_eval

from model.util import Normalizer, seed_everything
from model.database_util import (
    generate_column_min_max,
    sample_all_tables,
    generate_query_bitmaps,
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
    bs = 36
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
    # device = 'cpu'
    newpath = './results/full/cost/'
    to_predict = 'cost'
    max_workers = 10  # Limit the number of multiprocessing workers

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
    generate_column_min_max(
        db_params=DB_PARAMS,
        imdb_schema=imdb_schema,
        output_file=column_min_max_file,
        t2alias=t2alias,
        max_workers=args.max_workers,
        pool_minconn=1,
        pool_maxconn=args.max_workers  # Ensure pool_maxconn >= max_workers
    )

column_min_max_vals = load_column_min_max(column_min_max_file)
logging.info(f"Loaded column min-max values from '{column_min_max_file}'.")

# Initialize Normalizers
cost_norm = Normalizer(-3.61192, 12.290855)  # Example values, adjust as needed
card_norm = Normalizer(1, 100)  # Example values, adjust as needed

# Perform sampling per table
sample_dir = './data/imdb/sampled_data/'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

logging.info("Starting sampling of all tables.")
sample_all_tables(
    db_params=DB_PARAMS,
    imdb_schema=imdb_schema,
    sample_dir=sample_dir,
    num_samples=1000,
    max_workers=args.max_workers
)
logging.info("Completed sampling of all tables.")


def extract_query_info_from_plan(json_plan, query_id, alias2t):
    """
    Extract tables, joins, predicates, and cardinality from a query plan in JSON format.

    Args:
        json_plan (dict): A query plan in JSON format.
        query_id (int): The ID of the query being processed.
        alias2t (dict): Mapping from table aliases to table names.

    Returns:
        str: A formatted string similar to synthetic.csv
    """
    # Extract tables
    tables = set()
    joins = []
    predicates = []
    # Extract cardinality
    cardinality = json_plan.get('Actual Rows', json_plan.get('Plan Rows', 0))

    def parse_plan_node(node, parent_alias=None):
        alias = node.get('Alias', parent_alias)
        # Extract tables
        if 'Relation Name' in node:
            tables.add(node['Relation Name'])
            alias = node.get('Alias', node['Relation Name'])
            logging.debug(f"Query_id={query_id}: Detected table '{node['Relation Name']}' with alias '{alias}'.")

        # Process joins
        if 'Hash Cond' in node or 'Join Filter' in node:
            join_cond = node.get('Hash Cond', node.get('Join Filter'))
            if join_cond:
                joins.append(join_cond)
                logging.debug(f"Query_id={query_id}: Detected join condition: {join_cond}")

        # Process predicates
        conditions = []
        for cond_type in ['Filter', 'Index Cond', 'Recheck Cond']:
            if cond_type in node:
                conditions.append(node[cond_type])

        # Include full table name in predicates
        for cond in conditions:
            # Replace parentheses and split conditions
            cond_clean = cond.replace('(', '').replace(')', '')
            preds = cond_clean.split(' AND ')
            for pred in preds:
                parts = pred.strip().split(' ', 2)
                if len(parts) == 3:
                    col, op, val = parts
                    # Check if 'val' is a column name (contains '.')
                    if '.' in val:
                        # This is a join predicate, skip adding to predicates
                        # logging.warning(f"Predicate '{pred}' in query_id={query_id} could be a join condition. Skipping.")
                        continue
                    if '.' not in col:
                        if alias:
                            table = alias2t.get(alias)
                            if not table:
                                # logging.warning(f"Alias '{alias}' not found in alias2t mapping. Skipping predicate '{pred}'.")
                                continue
                            col = f"{table}.{col}"
                            logging.debug(f"Query_id={query_id}: Prefixed column '{col}' with table '{table}'.")
                        else:
                            logging.warning(f"Cannot determine alias for column '{col}' in query_id={query_id}. Skipping predicate.")
                            continue
                    predicates.append(f"({col} {op} {val})")
                else:
                    logging.warning(f"Incomplete predicate: '{pred}' in query_id={query_id}. Skipping.")

        # Recursively handle subplans
        if 'Plans' in node:
            for subplan in node['Plans']:
                parse_plan_node(subplan, parent_alias=alias)

    parse_plan_node(json_plan)

    # Join tables, joins, and predicates into the desired format
    table_str = ",".join(sorted(list(tables)))
    join_str = ",".join(joins) if joins else ""
    predicate_str = ",".join(predicates) if predicates else ""
    cardinality_str = str(cardinality)

    logging.debug(f"Query_id={query_id}: Extracted query info: tables={table_str}, joins={join_str}, predicates={predicate_str}, cardinality={cardinality_str}")

    return f"{table_str}#{join_str}#{predicate_str}#{cardinality_str}"


def generate_csv_for_samples(df, output_path, alias2t):
    """
    Generates a CSV similar to synthetic.csv based on a DataFrame (train_df or val_df).

    Args:
        df (pd.DataFrame): Data containing JSON query plans.
        output_path (str): Path to save the generated CSV.
        alias2t (dict): Mapping from table aliases to table names.
    """
    query_info_list = []

    for idx, row in df.iterrows():
        # Assuming 'json' column contains the query plan in JSON format
        try:
            plan_json = json.loads(row['json'])
            query_info = extract_query_info_from_plan(plan_json['Plan'], query_id=idx, alias2t=alias2t)
            query_info_list.append(query_info)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error for query_id={idx}: {e}. Skipping this query.")
            query_info_list.append("NA#NA#NA#0")  # Placeholder for failed parsing
        except KeyError as e:
            logging.error(f"Missing key {e} in query_id={idx}. Skipping this query.")
            query_info_list.append("NA#NA#NA#0")  # Placeholder for failed parsing

    # Save the query information to a CSV file
    with open(output_path, 'w') as f:
        for query_info in query_info_list:
            f.write(f"{query_info}\n")

    logging.info(f"CSV file saved to: {output_path}")




# Generate CSV for queries based on train_df and val_df
imdb_path = './data/imdb/'
train_parts = [f"{imdb_path}plan_and_cost/train_plan_part{i}.csv" for i in range(18)]  # Example: parts 0 and 1
dfs = []
for file in train_parts:
    if os.path.exists(file):
        df_part = pd.read_csv(file)
        dfs.append(df_part)
    else:
        logging.warning(f"Training part file '{file}' does not exist. Skipping.")
full_train_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

val_parts = [f"{imdb_path}plan_and_cost/train_plan_part{i}.csv" for i in range(18, 20)]
val_dfs = []
for file in val_parts:
    if os.path.exists(file):
        df_part = pd.read_csv(file)
        val_dfs.append(df_part)
    else:
        logging.warning(f"Validation part file '{file}' does not exist. Skipping.")
val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()

logging.info(f"Loaded training data with {len(full_train_df)} records.")
logging.info(f"Loaded validation data with {len(val_df)} records.")

combined_df = pd.concat([full_train_df, val_df], ignore_index=True) if not full_train_df.empty and not val_df.empty else pd.DataFrame()
# combined_df = combined_df.head(100)  # only fetch the first 100 queries for testing
generated_csv_path = './data/imdb/generated_queries.csv'

# Generate CSV for queries based on train_df and val_df
if not combined_df.empty:
    if os.path.exists(generated_csv_path):
        logging.warning(f"Generated CSV file '{generated_csv_path}' already exists. Skipping generation.")
    else:
        generate_csv_for_samples(combined_df, generated_csv_path, alias2t)
else:
    logging.error("Combined training and validation DataFrame is empty. Exiting.")
    exit(1)

# Load all queries from the generated CSV
column_names = ['tables', 'joins', 'predicate', 'cardinality']
try:
    query_file = pd.read_csv(
        generated_csv_path,
        sep='#',
        header=None,
        names=column_names,
        keep_default_na=False,   # Do not convert empty strings to NaN
        na_values=['']           # Treat empty strings as empty, not NaN
    )
    # only fetch the first 100 queries for testing
    # query_file = query_file.head(100)
except pd.errors.ParserError as e:
    logging.error(f"Error reading generated_queries.csv: {e}")
    exit(1)

# Generate bitmaps for each query based on pre-sampled table data
logging.info("Generating table sample bitmaps for each query.")

sampled_data = generate_query_bitmaps(
    query_file=query_file,
    alias2t=alias2t,
    sample_dir=sample_dir
)

logging.info("Completed generating table sample bitmaps for all queries.")

# Generate histograms based on entire tables
hist_dir = './data/imdb/histograms/'
histogram_file_path = './data/imdb/histogram_entire.csv'

if not os.path.exists(histogram_file_path):
    hist_file_df = generate_histograms_entire_db(
        db_params=DB_PARAMS,
        imdb_schema=imdb_schema,
        hist_dir=hist_dir,
        bin_number=50,
        t2alias=t2alias,
        max_workers=args.max_workers
    )
    # Save histograms with comma-separated bins
    save_histograms(hist_file_df, save_path=histogram_file_path)
else:
    hist_file_df = load_entire_histograms(load_path=histogram_file_path)

encoding = Encoding(column_min_max_vals=column_min_max_vals, col2idx=col2idx)
logging.info("Initialized Encoding object.")

# Seed for reproducibility
seed_everything()

# Initialize PlanTreeDataset
train_ds = PlanTreeDataset(
    json_df=combined_df,
    workload_df=None,  # Assuming workload_df is not needed for training
    encoding=encoding,
    hist_file=hist_file_df,
    card_norm=card_norm,
    cost_norm=cost_norm,
    to_predict=args.to_predict,
    table_sample=sampled_data,  # This should be a list indexed by query_id
    alias2t=alias2t,
)

val_ds = PlanTreeDataset(
    json_df=combined_df,
    workload_df=None,  # Assuming workload_df is not needed for validation
    encoding=encoding,
    hist_file=hist_file_df,
    card_norm=card_norm,
    cost_norm=cost_norm,
    to_predict=args.to_predict,
    table_sample=sampled_data,  # Ensure consistency
    alias2t=alias2t,
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
    'get_sample': lambda workload_file: generate_query_bitmaps(
        query_file=pd.read_csv(workload_file, sep='#', header=None, names=['tables', 'joins', 'predicate', 'cardinality'], keep_default_na=False, na_values=['']),
        alias2t=alias2t,
        sample_dir=sample_dir
    ),
    'encoding': encoding,
    'cost_norm': cost_norm,
    'hist_file': hist_file_df,
    'model': model,
    'device': args.device,
    'bs': 512,
}
exit()
# Evaluate on 'job-light' workload
job_light_workload_file = './data/imdb/workloads/job-light.csv'
if os.path.exists(job_light_workload_file):
    job_light_scores, job_light_corr = eval_workload('job-light', methods)
    logging.info(f"Job-Light Workload Evaluation: {job_light_scores}, Correlation: {job_light_corr}")
else:
    logging.warning(f"Job-Light workload file '{job_light_workload_file}' does not exist. Skipping evaluation.")

# Evaluate on 'synthetic' workload
synthetic_workload_file = './data/imdb/workloads/synthetic.csv'
if os.path.exists(synthetic_workload_file):
    synthetic_scores, synthetic_corr = eval_workload('synthetic', methods)
    logging.info(f"Synthetic Workload Evaluation: {synthetic_scores}, Correlation: {synthetic_corr}")
else:
    logging.warning(f"Synthetic workload file '{synthetic_workload_file}' does not exist. Skipping evaluation.")
