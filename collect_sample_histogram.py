# %%
import pandas as pd
import json
import re
import math
import numpy as np
from collections import defaultdict
import psycopg2
import time
import os

# %%
imdb_schema = {'title': ['t.id', 't.kind_id', 't.production_year'],
 'movie_companies': ['mc.id',
  'mc.company_id',
  'mc.movie_id',
  'mc.company_type_id'],
 'cast_info': ['ci.id', 'ci.movie_id', 'ci.person_id', 'ci.role_id'],
 'movie_info_idx': ['mi_idx.id', 'mi_idx.movie_id', 'mi_idx.info_type_id'],
 'movie_info': ['mi.id', 'mi.movie_id', 'mi.info_type_id'],
 'movie_keyword': ['mk.id', 'mk.movie_id', 'mk.keyword_id']}
t2alias = {'title':'t','movie_companies':'mc','cast_info':'ci',
          'movie_info_idx':'mi_idx','movie_info':'mi','movie_keyword':'mk'}
alias2t = {}
for k,v in t2alias.items(): alias2t[v] = k

# %%
conm = psycopg2.connect(database="imdb", user="wuy", host="127.0.0.1",password="wuy", port="5432")
conm.set_session(autocommit=True)
cur = conm.cursor()

# %%
def to_vals(data_list):
    '''
    convert a list of tuples to a numpy array of values
    '''
    for dat in data_list:
        val = dat[0]
        # finds first non-None value
        if val is not None: break 
    try:
        float(val)
        return np.array(data_list, dtype=float).squeeze()
    except:
#         print(val)
        res = []
        for dat in data_list:
            try:
                mi = dat[0].timestamp()
            except:
                mi = 0
            res.append(mi)
        return np.array(res)

# %% [markdown]
# ## Histogram

# %%
hist_file_path ='data/imdb/new_hist_file.csv'
if os.path.exists(hist_file_path):
    hist_file = pd.read_csv(hist_file_path)
else:
    hist_file = pd.DataFrame(columns=['table','column','bins','table_column'])
    hist_file
    for table,columns in imdb_schema.items():
        for column in columns:
            cmd = 'select {} from {} as {}'.format(column, table,t2alias[table])
            cur.execute(cmd)
            col = cur.fetchall()
            col_array = to_vals(col)
            # calculate the percentile of the col_array, at intervals of 2%, forming the basis for the histogram bins.
            hists = np.nanpercentile(col_array, range(0,101,2), axis=0)
            res_dict = {
                'table':table,
                'column':column,
                'table_column': '.'.join((table, column)),
                'bins':hists
            }
            # hist_file = hist_file.append(res_dict,ignore_index=True)
            hist_file = pd.concat([hist_file, pd.DataFrame(res_dict)], ignore_index=True)
            hist_file.to_csv(hist_file_path, index=False)

# %%


# %%
hist_file

# %%


# %%


# %%


# %% [markdown]
# ## Simpler Approach to sample without creating a smaller database

# %%
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
import os
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

NUM_SAMPLES_PER_TABLE = 1000  # Adjust as needed

def sample_table(cursor, table, alias, num_samples):
    """
    Samples a specified number of rows from a table using TABLESAMPLE SYSTEM_ROWS.

    Args:
        cursor: psycopg2 cursor object.
        table (str): Table name.
        alias (str): Alias for the table.
        num_samples (int): Number of samples to fetch.

    Returns:
        pd.DataFrame: Sampled data as a DataFrame.
    """
    try:
        # Calculate the sampling rate to approximate the desired number of samples
        # Note: SYSTEM_ROWS may not guarantee exact sample size; adjust as needed
        sampling_cmd = f"SELECT * FROM {table} AS {alias} TABLESAMPLE SYSTEM_ROWS({num_samples}) "
        cursor.execute(sampling_cmd)
        samples = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(samples, columns=colnames)
        logging.info(f"Sampled {len(df)} rows from table '{table}'.")
        return df
    except Exception as e:
        logging.error(f"Error sampling table '{table}': {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Dictionary to hold sampled data
sampled_data = {}

for table, columns in imdb_schema.items():
    alias = t2alias[table]
    df = sample_table(cur, table, alias, NUM_SAMPLES_PER_TABLE)
    if not df.empty:
        # Assign a unique sample ID
        df['sid'] = range(len(df))
        sampled_data[table] = df



# %% [markdown]
# ## Sample
# ### Steps (There may be other easier methods)
# 1. generate 1000 sample points for each table
# 2. duplicate database schema from full db
#     > pg_dump imdb -s -O > imdb_schema.sql
# 3. create small base by in psql
#     > create database imdb_sample
# 4. create schema using imdb_schema.sql
# 5. load the sample data using pandas and sqlalchemy
# 6. query the small base to get sample bitmaps for each predicate

# %%


# %%


# %% [markdown]
# Step 1

# %%
## sampling extension
cmd = 'CREATE EXTENSION tsm_system_rows' # enable system rows sampling
cur.execute(cmd)

# %%
tables = list(imdb_schema.keys())
sample_data = {}
for table in tables:
    cur.execute("Select * FROM {} LIMIT 0".format(table))
    colnames = [desc[0] for desc in cur.description] # cur.description provides metadata about the result set, including column names and data types.

    ts = pd.DataFrame(columns = colnames)

    for num in range(1000):
        cmd = 'SELECT * FROM {} TABLESAMPLE SYSTEM_ROWS(1)'.format(table) # return 1 row per table
        cur.execute(cmd)
        samples = cur.fetchall()
        for i,row in enumerate(samples):
            ts.loc[num]=row
    
    sample_data[table] = ts

# %%
sample_data['title']

# %% [markdown]
# Step 5 (Do step 2-4 outside first)

# %%
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:admin@localhost:5432/imdb_sample')

# %%
for k,v in sample_data.items():
    v['sid'] = list(range(1000))
    cmd = 'alter table {} add column sid integer'.format(k)
    cur.execute(cmd)
    v.to_sql(k,engine,if_exists='append',index=False) # insert the DataFrame `v` into the table `k` within the database `imdb_sample`

# %%


# %% [markdown]
# Step 6

# %%


# %%
query_file = pd.read_csv('data/imdb/workloads/synthetic.csv',sep='#',header=None)
query_file.columns = ['table','join','predicate','card']

# %%
query_file.head()

# %%
conm = psycopg2.connect(database="imdb_sample", user="postgres", host="127.0.0.1",password="admin", port="5432")
conm.set_session(autocommit=True)
cur = conm.cursor()

# %%


# %%
table_samples = []
for i,row in query_file.iterrows():
    table_sample = {}
    preds = row['predicate'].split(',')
    for i in range(0,len(preds),3):
        left, op, right = preds[i:i+3]
        alias,col = left.split('.')
        table = alias2t[alias]
        pred_string = ''.join((col,op,right))
        # Constructs a SQL query to select the `sid` (sample ID) from the current `table` where the predicate holds true.
        q = 'select sid from {} where {}'.format(table, pred_string)
        cur.execute(q)
        sps = np.zeros(1000).astype('uint8')
        sids = cur.fetchall()
        sids = np.array(sids).squeeze()
        if sids.size>1: # at least 1 sample satisfies the predicate
            sps[sids] = 1
        if table in table_sample:
            table_sample[table] = table_sample[table] & sps
        else:
            table_sample[table] = sps
    table_samples.append(table_sample)

# %%
# table_samples

# %%



