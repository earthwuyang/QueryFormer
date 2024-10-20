# # %%
# %load_ext autoreload
# %autoreload 2


# %%
import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr

# %%
from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample, collator
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train

# %%
dataset = 'tpcds_sf1'
tmp_data_dir = 'data/tpcds_sf1'
import json

# column_type_file = os.path.join(os.path.dirname(__file__), f'../zsce/cross_db_benchmark/datasets/{dataset}/column_type.json')
column_type_file = os.path.join(f'/home/wuy/DB/memory_prediction/zsce/cross_db_benchmark/datasets/{dataset}/column_type.json')
with open(column_type_file, 'r') as f:
    column_type = json.load(f)

schema = {}
for table, columns in column_type.items():
    if table == 'dbgen_version':
        continue
    for column, type_ in columns.items():
        if table not in schema:
            schema[table] = []
        schema[table].append(column)

# Define table aliases as their original names for tpcds by iterating tpcds_schema, meanwhile get col2idx
t2alias = {}
col2idx = {}
for table, columns in schema.items():
    for column in columns:
        t2alias[table] = table
        col2idx[table + '.' + column] = len(col2idx)
# add 'NA' to the last of col2idx
col2idx['NA'] = len(col2idx)

alias2t = {v: k for k, v in t2alias.items()}

# %%
import os
import logging
import multiprocessing
import psycopg2
import pandas as pd
from tqdm import tqdm

DB_PARAMS = {
    'database': 'tpcds_sf1',
    'user': "wuy",
    'host': "127.0.0.1",
    'password': "wuy",
    'port': "5432"
}

def extract_column_stats(args):
    """
    Extracts min, max, cardinality, and number of unique values for a single table-column pair.
    
    Args:
        args (tuple): Contains (table, column, db_params, t2alias).
    
    Returns:
        dict or None: Dictionary with column statistics or None if an error occurs.
    """
    table, column, db_params, t2alias = args
    stats = {}
    
    # Skip 'sid' column if present
    if column == 'sid':
        return None
    
    try:
        # Establish a new database connection for each subprocess
        conn = psycopg2.connect(**db_params)
        conn.set_session(autocommit=True)
        cur = conn.cursor()
        
        # Construct SQL queries
        min_query = f"SELECT MIN({column}) FROM {table};"
        max_query = f"SELECT MAX({column}) FROM {table};"
        count_query = f"SELECT COUNT({column}) FROM {table} WHERE {column} IS NOT NULL;"
        distinct_query = f"SELECT COUNT(DISTINCT {column}) FROM {table} WHERE {column} IS NOT NULL;"
        
        # Execute queries and fetch results
        cur.execute(min_query)
        min_val = cur.fetchone()[0]
        
        cur.execute(max_query)
        max_val = cur.fetchone()[0]
        
        cur.execute(count_query)
        cardinality = cur.fetchone()[0]
        
        cur.execute(distinct_query)
        num_unique = cur.fetchone()[0]
        
        # Populate the stats dictionary
        stats = {
            'name': f"{t2alias.get(table, table[:2])}.{column}",
            'min': min_val,
            'max': max_val,
            'cardinality': cardinality,
            'num_unique_values': num_unique
        }
        
        # logging.info(f"Extracted stats for '{table}.{column}': min={min_val}, max={max_val}, cardinality={cardinality}, unique={num_unique}")
        
        # Close the cursor and connection
        cur.close()
        conn.close()
        
        return stats
    
    except Exception as e:
        logging.error(f"Error extracting stats for '{table}.{column}': {e}")
        return None
    

def generate_column_min_max(db_params, schema, output_file, t2alias={}, max_workers=10, pool_minconn=1, pool_maxconn=10):
    """
    Connects to the PostgreSQL database, extracts min, max, cardinality, and number of unique values
    for each column in the specified tables using multiprocessing, and saves the statistics to a CSV file.
    
    Args:
        db_params (dict): Database connection parameters.
        tpcds_schema (dict): Schema dictionary mapping table names to their columns.
        output_file (str): Path to save the generated CSV file.
        t2alias (dict): Table aliases.
        max_workers (int): Maximum number of multiprocessing workers.
        pool_minconn (int): Minimum number of connections in the pool.
        pool_maxconn (int): Maximum number of connections in the pool.
    
    Returns:
        None
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    
    # List to hold table-column pairs
    table_column_pairs = []
    for table, columns in schema.items():
        for column in columns:
            if column == 'sid':
                continue  # Skip 'sid' column
            table_column_pairs.append((table, column, db_params, t2alias))
    
    # Determine the number of worker processes
    num_workers = min(len(table_column_pairs), max_workers)
    logging.info(f"Starting multiprocessing with {num_workers} workers for {len(table_column_pairs)} table-column pairs.")
    
    # Initialize a multiprocessing Pool with limited workers
    with multiprocessing.Pool(processes=num_workers) as pool_mp:
        # Use imap_unordered for better performance and to handle results as they come
        results = []
        for res in tqdm(pool_mp.imap_unordered(extract_column_stats, table_column_pairs), total=len(table_column_pairs)):
            if res is not None:
                results.append(res)
    
    # Create a DataFrame from the results
    stats_df = pd.DataFrame(results)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the DataFrame to CSV
    stats_df.to_csv(output_file, index=False)
    logging.info(f"Saved column statistics to '{output_file}'.")

# %%
column_min_max_path = 'data/tpcds_sf1/column_min_max.csv'
if not os.path.exists(column_min_max_path):
    generate_column_min_max(DB_PARAMS, schema, 'data/tpcds_sf1/column_min_max.csv', t2alias)
column_min_max_vals = pd.read_csv(column_min_max_path)


# %%

dataset = 'tpcds_sf1'
tmp_data_dir = 'data/tpcds_sf1'

# %%
class Args:
    # bs = 1024
    # SQ: smaller batch size
    bs = 512
    lr = 0.001
    epochs = 200
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cuda:0'
    # device = 'cpu'
    newpath = './results/mem/'
    to_predict = 'mem'
args = Args()

import os
if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)

# %%
hist_file = get_hist_file(os.path.join(tmp_data_dir, 'hist_file.csv'))
data_dir = '/home/wuy/DB/pg_mem_data'
with open(os.path.join(data_dir, dataset, 'statistics_workload_combined.json')) as f:
        stats = json.load(f)
max_label = stats['peakmem']['max']
scale_label = stats['peakmem']['scale']
center_label = stats['peakmem']['center']
from sklearn.preprocessing import RobustScaler
label_norm = RobustScaler()

label_norm.scale_ = np.array([scale_label])
label_norm.center_ = np.array([center_label])


# %%
# encoding_ckpt = torch.load('checkpoints/encoding.pt')
# encoding = encoding_ckpt['encoding']
from model.database_util import Encoding
import pickle
encoding_file = os.path.join(tmp_data_dir, 'encoding.pkl')
if os.path.exists(encoding_file):
    with open(encoding_file, 'rb') as f:
        encoding = pickle.load(f)
else:
    encoding = Encoding(column_min_max_vals, col2idx)

# checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu')

# %%
from model.util import seed_everything
seed_everything()

# %%
to_predict = 'cost'

# %%
import json
train_plans_file = '/home/wuy/DB/pg_mem_data/tpcds_sf1/train_plans.json'
with open(train_plans_file, 'r') as f:
    train_plans = json.load(f)

val_plans_file = '/home/wuy/DB/pg_mem_data/tpcds_sf1/val_plans.json'
with open(val_plans_file, 'r') as f:
    val_plans = json.load(f)

test_plans_file = '/home/wuy/DB/pg_mem_data/tpcds_sf1/test_plans.json'
with open(test_plans_file, 'r') as f:
    test_plans = json.load(f)

# %%
# import pickle
# # table_sample = get_job_table_sample(plans, 'data/tpcds_sf1/sample_data.pkl')
# with open('data/tpcds_sf1/table_samples.pkl', 'rb') as f:
#     table_sample = pickle.load(f)

# %%

train_ds = PlanTreeDataset(train_plans, None, encoding, hist_file, label_norm, to_predict, 'train', column_type, DB_PARAMS)
val_ds = PlanTreeDataset(val_plans, None, encoding, hist_file, label_norm, to_predict, 'val', column_type, DB_PARAMS)
test_ds = PlanTreeDataset(test_plans, None, encoding, hist_file, label_norm, to_predict, 'test', column_type, DB_PARAMS)


if not os.path.exists(encoding_file):
    with open(encoding_file, 'wb') as f:
        pickle.dump(encoding, f)
        
# %%
model = QueryFormer(emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, 
                 dropout = args.dropout, n_layers = args.n_layers, 
                 use_sample = True, use_hist = True, 
                 pred_hid = args.pred_hid, joins = len(encoding.join2idx), tables = len(encoding.table2idx), types = len(encoding.type2idx), columns = len(encoding.col2idx), 
                )
_ = model.to(args.device)

# %%
crit = nn.MSELoss()
model, best_path = train(model, train_ds, val_ds, test_ds, crit, label_norm, args)

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



