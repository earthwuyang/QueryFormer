# model/database_util.py

import numpy as np
import pandas as pd
import csv
import torch
import psycopg2
import logging
import os
import ast
import multiprocessing
from tqdm import tqdm

## BFS should be enough
def floyd_warshall_rewrite(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j: 
                M[i][j] = 0
            elif M[i][j] == 0: 
                M[i][j] = 60
    
    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k]+M[k][j])
    return M


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
        
        logging.info(f"Extracted stats for '{table}.{column}': min={min_val}, max={max_val}, cardinality={cardinality}, unique={num_unique}")
        
        # Close the cursor and connection
        cur.close()
        conn.close()
        
        return stats
    
    except Exception as e:
        logging.error(f"Error extracting stats for '{table}.{column}': {e}")
        return None
    

def generate_column_min_max(db_params, imdb_schema, output_file='./data/imdb/column_min_max_vals.csv', t2alias={}, max_workers=4, pool_minconn=1, pool_maxconn=10):
    """
    Connects to the PostgreSQL database, extracts min, max, cardinality, and number of unique values
    for each column in the specified tables using multiprocessing, and saves the statistics to a CSV file.
    
    Args:
        db_params (dict): Database connection parameters.
        imdb_schema (dict): Schema dictionary mapping table names to their columns.
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
    for table, columns in imdb_schema.items():
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


def sample_table(args):
    """
    Samples a specified number of rows from a table and saves them to a CSV file.
    
    Args:
        args (tuple): Contains (table, db_params, sample_dir, num_samples).
    
    Returns:
        str: Path to the sampled CSV file or None if an error occurs.
    """
    table, db_params, sample_dir, num_samples = args
    sample_file = os.path.join(sample_dir, f"{table}_sampled.csv")
    
    try:
        # Establish a new database connection for each subprocess
        conn = psycopg2.connect(**db_params)
        conn.set_session(autocommit=True)
        cur = conn.cursor()
        
        # Construct SQL query to sample rows
        # Using TABLESAMPLE SYSTEM for random sampling (adjust method as needed)
        # Alternatively, use ORDER BY RANDOM() LIMIT num_samples for exact sampling
        query = f"SELECT * FROM {table} TABLESAMPLE SYSTEM ({(num_samples / 100000) * 100}) LIMIT {num_samples};"
        # Note: Adjust the sampling percentage based on table size
        
        cur.execute(query)
        sampled_rows = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]
        
        # Save sampled rows to CSV
        df = pd.DataFrame(sampled_rows, columns=column_names)
        df.to_csv(sample_file, index=False)
        
        logging.info(f"Sampled {len(df)} rows from '{table}' and saved to '{sample_file}'.")
        
        # Close the cursor and connection
        cur.close()
        conn.close()
        
        return sample_file
    
    except Exception as e:
        logging.error(f"Error sampling table '{table}': {e}")
        return None


def sample_all_tables(db_params, imdb_schema, sample_dir='./data/imdb/sampled_data/', num_samples=1000, max_workers=4):
    """
    Samples data for each table using multiprocessing.
    
    Args:
        db_params (dict): Database connection parameters.
        imdb_schema (dict): Schema dictionary mapping table names to their columns.
        sample_dir (str): Directory to save sampled CSV files.
        num_samples (int): Number of samples per table.
        max_workers (int): Maximum number of multiprocessing workers.
    
    Returns:
        dict: Dictionary mapping table names to their sampled DataFrame.
    """
    # List of tables to sample
    tables = list(imdb_schema.keys())
    
    # Prepare arguments for each table
    args_list = [
        (table, db_params, sample_dir, num_samples)
        for table in tables
    ]
    
    # Determine the number of worker processes
    num_workers = min(len(tables), max_workers)
    logging.info(f"Starting multiprocessing with {num_workers} workers for sampling {len(tables)} tables.")
    
    # Initialize a multiprocessing Pool with limited workers
    with multiprocessing.Pool(processes=num_workers) as pool_mp:
        # Use imap_unordered with tqdm for progress tracking
        sampled_files = []
        for sample_file in tqdm(pool_mp.imap_unordered(sample_table, args_list), total=len(args_list)):
            if sample_file is not None:
                sampled_files.append(sample_file)
    
    # Load sampled data into a dictionary
    sampled_data = {}
    for sample_file in sampled_files:
        table = os.path.basename(sample_file).replace('_sampled.csv', '')
        df = pd.read_csv(sample_file)
        sampled_data[table] = df
    
    return sampled_data


def generate_query_bitmaps(query_file, alias2t, sample_dir='./data/imdb/sampled_data/'):
    """
    Generates table sample bitmaps for each query based on pre-sampled table data.
    
    Args:
        query_file (pd.DataFrame): DataFrame containing queries.
        alias2t (dict): Mapping from table aliases to table names.
        sample_dir (str): Directory where sampled table CSV files are stored.
    
    Returns:
        list: List of dictionaries containing bitmaps for each query's tables.
    """
    table_sample_bitmaps = []
    
    for idx, row in tqdm(query_file.iterrows(), total=len(query_file)):
        query_id = row['id']
        tables_joins = row['tables_joins'].split(',')
        predicates = row['predicate'].split(',')
        
        # Initialize bitmap dictionary for this query
        bitmap_dict = {}
        
        # Parse tables and aliases from 'tables_joins'
        involved_tables = set()
        for tj in tables_joins:
            alias, table = tj.split('.')
            involved_tables.add(alias2t.get(alias, table))
        
        # Load sampled data for involved tables
        sampled_tables = {}
        for table in involved_tables:
            sample_file = os.path.join(sample_dir, f"{table}_sampled.csv")
            if os.path.exists(sample_file):
                df = pd.read_csv(sample_file)
                sampled_tables[table] = df
            else:
                logging.warning(f"Sample file '{sample_file}' does not exist for table '{table}'.")
                sampled_tables[table] = pd.DataFrame()  # Empty DataFrame
        
        # Apply predicates to generate bitmaps
        # Assuming predicates are in the format: "alias.column,operator,value"
        # e.g., "t.production_year,<,2020"
        num_tables = len(involved_tables)
        for i in range(0, len(predicates), 3):
            if i + 2 >= len(predicates):
                break  # Incomplete predicate
            left, op, right = predicates[i:i+3]
            alias, column = left.split('.')
            table = alias2t.get(alias, alias)  # Default to alias if not found
            df = sampled_tables.get(table, pd.DataFrame())
            if df.empty:
                # If no data, default bitmap to all zeros
                bitmap = np.zeros(len(df), dtype='uint8')
            else:
                if op == '=':
                    bitmap = (df[column] == float(right)).astype('uint8').values
                elif op == '<':
                    bitmap = (df[column] < float(right)).astype('uint8').values
                elif op == '>':
                    bitmap = (df[column] > float(right)).astype('uint8').values
                else:
                    logging.warning(f"Unsupported operator '{op}' in predicate.")
                    bitmap = np.zeros(len(df), dtype='uint8')
            bitmap_dict[f"{table}.{column}"] = bitmap
        
        table_sample_bitmaps.append(bitmap_dict)
    
    return table_sample_bitmaps


def generate_histogram_single(args):
    """
    Generates histogram for a single table-column pair.

    Args:
        args (tuple): Contains (table, column, db_params, hist_dir, bin_number, t2alias).

    Returns:
        dict: Histogram record for the table-column.
    """
    table, column, db_params, hist_dir, bin_number, t2alias = args

    if column == 'sid':
        return None  # Skip the sample ID column

    hist_file = os.path.join(hist_dir, f"{table}_{column}_histogram.csv")

    if os.path.exists(hist_file):
        try:
            df = pd.read_csv(hist_file)
            bins = df['bins'].tolist()
            logging.info(f"Loaded histogram for '{table}.{column}' from '{hist_file}'.")
        except Exception as e:
            logging.error(f"Error loading histogram from '{hist_file}': {e}")
            bins = []
    else:
        try:
            # Establish a separate database connection for each subprocess
            conn = psycopg2.connect(**db_params)
            conn.set_session(autocommit=True)
            cur = conn.cursor()

            query = f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"
            cur.execute(query)
            data = cur.fetchall()
            data = [row[0] for row in data if row[0] is not None]
            if not data:
                logging.warning(f"No data found for histogram generation for '{table}.{column}'.")
                cur.close()
                conn.close()
                return None

            # Compute percentiles as bin edges
            bins = np.percentile(data, np.linspace(0, 100, bin_number + 1))

            # Save histogram to CSV
            pd.DataFrame({'bins': bins.tolist()}).to_csv(hist_file, index=False)
            logging.info(f"Generated and saved histogram for '{table}.{column}' to '{hist_file}'.")

            cur.close()
            conn.close()
        except Exception as e:
            logging.error(f"Error generating histogram for '{table}.{column}': {e}")
            bins = []

    if len(bins) != 0:
        return {
            'table': table,
            'column': column,
            'bins': bins,
            'table_column': f"{t2alias.get(table, table[:2])}.{column}"
        }
    else:
        return None


def generate_histograms_entire_db(db_params, imdb_schema, hist_dir='./data/imdb/histograms/', bin_number=50, t2alias={}, max_workers=4):
    """
    Generates histograms for each column based on the entire data of each table using multiprocessing.
    Histograms are saved to CSV files.

    Args:
        db_params (dict): Database connection parameters.
        imdb_schema (dict): Schema dictionary mapping table names to their columns.
        hist_dir (str): Directory to save histogram CSV files.
        bin_number (int): Number of bins for the histograms.
        t2alias (dict): Mapping from table aliases to table names.
        max_workers (int): Maximum number of multiprocessing workers.

    Returns:
        pd.DataFrame: DataFrame containing histograms for each table.column.
    """
    os.makedirs(hist_dir, exist_ok=True)
    hist_records = []

    # Prepare arguments for each table-column pair
    args_list = []
    for table, columns in imdb_schema.items():
        for column in columns:
            if column == 'sid':
                continue  # Skip the sample ID column
            args_list.append((table, column, db_params, hist_dir, bin_number, t2alias))

    # Determine the number of worker processes
    num_workers = min(len(args_list), max_workers)
    logging.info(f"Starting multiprocessing with {num_workers} workers for generating histograms.")

    # Initialize a multiprocessing Pool with limited workers
    with multiprocessing.Pool(processes=num_workers) as pool_mp:
        # Use imap_unordered with tqdm for progress tracking
        for record in tqdm(pool_mp.imap_unordered(generate_histogram_single, args_list), total=len(args_list)):
            if record is not None:
                hist_records.append(record)

    # Create a DataFrame from the results
    hist_file_df = pd.DataFrame(hist_records)

    return hist_file_df


def save_histograms(hist_file_df, save_path='./data/imdb/histogram_entire.csv'):
    """
    Saves the histograms DataFrame to a CSV file.

    Args:
        hist_file_df (pd.DataFrame): DataFrame containing histograms.
        save_path (str): Path to save the histogram CSV.
    """
    hist_file_df.to_csv(save_path, index=False)
    logging.info(f"Saved entire table histograms to '{save_path}'.")


def load_entire_histograms(load_path='./data/imdb/histogram_entire.csv'):
    """
    Loads the histograms DataFrame from a CSV file.

    Args:
        load_path (str): Path to load the histogram CSV.

    Returns:
        pd.DataFrame: DataFrame containing histograms.
    """
    if not os.path.exists(load_path):
        logging.error(f"Histogram file '{load_path}' does not exist.")
        raise FileNotFoundError(f"Histogram file '{load_path}' does not exist.")
    hist_file_df = pd.read_csv(load_path)
    # Safely parse the 'bins' column
    hist_file_df['bins'] = hist_file_df['bins'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    logging.info(f"Loaded entire table histograms from '{load_path}'.")
    return hist_file_df


def sample_single_query(args):
    """
    This function is no longer needed in the two-step sampling approach.
    Retained for reference but can be removed.
    """
    pass  # Implemented in 'generate_query_bitmaps'


def generate_query_bitmaps(query_file, alias2t, sample_dir='./data/imdb/sampled_data/'):
    """
    Generates table sample bitmaps for each query based on pre-sampled table data.

    Args:
        query_file (pd.DataFrame): DataFrame containing queries.
        alias2t (dict): Mapping from table aliases to table names.
        sample_dir (str): Directory where sampled table CSV files are stored.

    Returns:
        list: List of dictionaries containing bitmaps for each query's tables.
    """
    table_sample_bitmaps = []

    for idx, row in tqdm(query_file.iterrows(), total=len(query_file)):
        query_id = row['id']
        
        # Handle NaN values in 'tables_joins'
        tables_joins = row['tables_joins']
        if pd.isnull(tables_joins):
            logging.warning(f"Query ID {query_id} has NaN in 'tables_joins'. Assigning empty list.")
            tables_joins = []
        else:
            try:
                tables_joins = tables_joins.split(',')
            except AttributeError:
                logging.warning(f"Query ID {query_id} has invalid type for 'tables_joins'. Assigning empty list.")
                tables_joins = []
        
        # Handle NaN values in 'predicate'
        predicates = row['predicate']
        if pd.isnull(predicates):
            logging.warning(f"Query ID {query_id} has NaN in 'predicate'. Assigning empty list.")
            predicates = []
        else:
            try:
                predicates = predicates.split(',')
            except AttributeError:
                logging.warning(f"Query ID {query_id} has invalid type for 'predicate'. Assigning empty list.")
                predicates = []
        
        # Initialize bitmap dictionary for this query
        bitmap_dict = {}

        # Parse tables and aliases from 'tables_joins'
        involved_tables = set()
        for tj in tables_joins:
            if '.' not in tj:
                logging.warning(f"Invalid 'tables_joins' format '{tj}' in query_id={query_id}. Skipping this entry.")
                continue
            alias, table = tj.split('.')
            actual_table = alias2t.get(alias, table)
            involved_tables.add(actual_table)

        # Load sampled data for involved tables
        sampled_tables = {}
        for table in involved_tables:
            sample_file = os.path.join(sample_dir, f"{table}_sampled.csv")
            if os.path.exists(sample_file):
                try:
                    df = pd.read_csv(sample_file)
                    sampled_tables[table] = df
                except Exception as e:
                    logging.error(f"Error reading sample file '{sample_file}' for table '{table}': {e}")
                    sampled_tables[table] = pd.DataFrame()  # Empty DataFrame
            else:
                logging.warning(f"Sample file '{sample_file}' does not exist for table '{table}'. Assigning empty DataFrame.")
                sampled_tables[table] = pd.DataFrame()  # Empty DataFrame

        # Apply predicates to generate bitmaps
        # Assuming predicates are in the format: "alias.column,operator,value"
        # e.g., "t.production_year,<,2020"
        num_predicates = len(predicates)
        for i in range(0, num_predicates, 3):
            if i + 2 >= num_predicates:
                logging.warning(f"Incomplete predicate in query_id={query_id}: {predicates[i:]}. Skipping this predicate.")
                break  # Incomplete predicate
            left, op, right = predicates[i:i+3]
            if '.' not in left:
                logging.warning(f"Invalid predicate format '{left}' in query_id={query_id}. Skipping this predicate.")
                continue
            alias, column = left.split('.')
            table = alias2t.get(alias, alias)
            df = sampled_tables.get(table, pd.DataFrame())
            if df.empty:
                # If no data, default bitmap to all zeros
                bitmap = np.zeros(0, dtype='uint8')
                logging.warning(f"No sampled data available for table '{table}' in query_id={query_id}. Assigning empty bitmap.")
            else:
                try:
                    if op == '=':
                        bitmap = (df[column] == float(right)).astype('uint8').values
                    elif op == '<':
                        bitmap = (df[column] < float(right)).astype('uint8').values
                    elif op == '>':
                        bitmap = (df[column] > float(right)).astype('uint8').values
                    else:
                        logging.warning(f"Unsupported operator '{op}' in predicate of query_id={query_id}. Assigning all ones.")
                        bitmap = np.ones(len(df), dtype='uint8')  # Assign all ones for unsupported operators
                except Exception as e:
                    logging.error(f"Error applying predicate '{left} {op} {right}' on table '{table}' for query_id={query_id}: {e}")
                    bitmap = np.zeros(len(df), dtype='uint8')  # Assign all zeros on error
            bitmap_dict[f"{table}.{column}"] = bitmap

        table_sample_bitmaps.append(bitmap_dict)

    return table_sample_bitmaps



def get_job_table_sample_direct(db_params, imdb_schema, query_file, alias2t, num_samples=1000, sample_dir='./data/imdb/sampled_data/', max_workers=4, t2alias={}):
    """
    Samples data directly from the IMDb database using multiprocessing.

    Args:
        db_params (dict): Database connection parameters.
        imdb_schema (dict): Schema dictionary mapping table names to their columns.
        query_file (pd.DataFrame): DataFrame containing queries.
        alias2t (dict): Mapping from table aliases to table names.
        num_samples (int): Number of samples to fetch per table.
        sample_dir (str): Directory to save/load sampled CSV files.
        max_workers (int): Maximum number of multiprocessing workers.
        t2alias (dict): Table aliases.

    Returns:
        list: List of table_sample dictionaries indexed by query_id.
    """
    os.makedirs(sample_dir, exist_ok=True)
    table_samples = []

    # Prepare arguments for each query
    args_list = [
        (row, alias2t, db_params, sample_dir, num_samples)
        for idx, row in query_file.iterrows()
    ]

    # Determine the number of worker processes
    num_workers = min(len(args_list), max_workers)
    logging.info(f"Starting multiprocessing with {num_workers} workers for sampling {len(args_list)} queries.")

    # Initialize a multiprocessing Pool with limited workers
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance and to handle results as they come
        for result in tqdm(pool.imap_unordered(sample_single_query, args_list), total=len(args_list)):
            table_samples.append(result)

    return table_samples


def generate_histogram_single(args):
    """
    Generates histogram for a single table-column pair.

    Args:
        args (tuple): Contains (table, column, db_params, hist_dir, bin_number, t2alias).

    Returns:
        dict: Histogram record for the table-column.
    """
    table, column, db_params, hist_dir, bin_number, t2alias = args

    if column == 'sid':
        return None  # Skip the sample ID column

    hist_file = os.path.join(hist_dir, f"{table}_{column}_histogram.csv")

    if os.path.exists(hist_file):
        try:
            df = pd.read_csv(hist_file)
            bins = df['bins'].tolist()
            logging.info(f"Loaded histogram for '{table}.{column}' from '{hist_file}'.")
        except Exception as e:
            logging.error(f"Error loading histogram from '{hist_file}': {e}")
            bins = []
    else:
        try:
            # Establish a separate database connection for each subprocess
            conn = psycopg2.connect(**db_params)
            conn.set_session(autocommit=True)
            cur = conn.cursor()

            query = f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"
            cur.execute(query)
            data = cur.fetchall()
            data = [row[0] for row in data if row[0] is not None]
            if not data:
                logging.warning(f"No data found for histogram generation for '{table}.{column}'.")
                cur.close()
                conn.close()
                return None

            # Compute percentiles as bin edges
            bins = np.percentile(data, np.linspace(0, 100, bin_number + 1))

            # Save histogram to CSV
            pd.DataFrame({'bins': bins.tolist()}).to_csv(hist_file, index=False)
            logging.info(f"Generated and saved histogram for '{table}.{column}' to '{hist_file}'.")

            cur.close()
            conn.close()
        except Exception as e:
            logging.error(f"Error generating histogram for '{table}.{column}': {e}")
            bins = []

    if len(bins) != 0:
        return {
            'table': table,
            'column': column,
            'bins': bins,
            'table_column': f"{t2alias.get(table, table[:2])}.{column}"
        }
    else:
        return None


def generate_histograms_entire_db(db_params, imdb_schema, hist_dir='./data/imdb/histograms/', bin_number=50, t2alias={}, max_workers=4):
    """
    Generates histograms for each column based on the entire data of each table using multiprocessing.
    Histograms are saved to CSV files.

    Args:
        db_params (dict): Database connection parameters.
        imdb_schema (dict): Schema dictionary mapping table names to their columns.
        hist_dir (str): Directory to save histogram CSV files.
        bin_number (int): Number of bins for the histograms.
        t2alias (dict): Mapping from table aliases to table names.
        max_workers (int): Maximum number of multiprocessing workers.

    Returns:
        pd.DataFrame: DataFrame containing histograms for each table.column.
    """
    os.makedirs(hist_dir, exist_ok=True)
    hist_records = []

    # Prepare arguments for each table-column pair
    args_list = []
    for table, columns in imdb_schema.items():
        for column in columns:
            if column == 'sid':
                continue  # Skip the sample ID column
            args_list.append((table, column, db_params, hist_dir, bin_number, t2alias))

    # Determine the number of worker processes
    num_workers = min(len(args_list), max_workers)
    logging.info(f"Starting multiprocessing with {num_workers} workers for generating histograms.")

    # Initialize a multiprocessing Pool with limited workers
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance and to handle results as they come
        for record in tqdm(pool.imap_unordered(generate_histogram_single, args_list), total=len(args_list)):
            if record is not None:
                hist_records.append(record)

    # Create a DataFrame from the results
    hist_file_df = pd.DataFrame(hist_records)

    return hist_file_df


def re_bin(hist_file, target_number):
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        bins = freq2bin(freq, target_number)
        hist_file['bins'][i] = bins
    return hist_file


def freq2bin(freqs, target_number):
    freq = freqs.copy()
    maxi = len(freq)-1
    
    step = 1. / target_number
    mini = 0
    while mini + 1 < len(freq) and freq[mini+1] == 0:
        mini += 1
    pointer = mini + 1
    cur_sum = 0
    res_pos = [mini]
    residue = 0
    while pointer < maxi + 1:
        cur_sum += freq[pointer]
        freq[pointer] = 0
        if cur_sum >= step:
            cur_sum -= step
            res_pos.append(pointer)
        else:
            pointer += 1
    
    if len(res_pos) == target_number:
        res_pos.append(maxi)
    
    return res_pos


class Batch():
    def __init__(self, attn_bias, rel_pos, heights, x, y=None):
        super(Batch, self).__init__()

        self.heights = heights
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.rel_pos = rel_pos
        
    def to(self, device):

        self.heights = self.heights.to(device)
        self.x = self.x.to(device)

        self.attn_bias, self.rel_pos = self.attn_bias.to(device), self.rel_pos.to(device)

        return self


def collator(small_set):
    """
    Collates a list of samples into a Batch object.
    """
    batch_dicts = [s[0] for s in small_set]
    y = [s[1] for s in small_set]
    xs = [s['x'] for s in batch_dicts]
    attn_bias = [s['attn_bias'] for s in batch_dicts]
    rel_pos = [s['rel_pos'] for s in batch_dicts]
    heights = [s['heights'] for s in batch_dicts]
    
    # Concatenate tensors
    x = torch.cat(xs, dim=0)
    attn_bias = torch.cat(attn_bias, dim=0)
    rel_pos = torch.cat(rel_pos, dim=0)
    heights = torch.cat(heights, dim=0)
    
    return Batch(attn_bias, rel_pos, heights, x), y


def filterDict2Hist(hist_file_df, filterDict, encoding):
    """
    Converts filter dictionaries to histogram embeddings using entire table histograms.
    
    Args:
        hist_file_df (pd.DataFrame): DataFrame containing histogram bins.
        filterDict (dict): Dictionary containing filter information.
        encoding (Encoding): Encoding object with necessary mappings.
    
    Returns:
        np.ndarray: Flattened histogram features.
    """
    bins_per_filter = len(hist_file_df['bins'].iloc[0]) - 1
    num_filters = len(filterDict['colId'])
    res = np.zeros((num_filters, bins_per_filter))
    
    for i in range(num_filters):
        colId = filterDict['colId'][i]
        col = encoding.idx2col.get(colId, 'NA')
        if col == 'NA':
            continue
        hist_row = hist_file_df[hist_file_df['table_column'] == col]
        if hist_row.empty:
            logging.warning(f"No histogram found for column '{col}'. Using zero histogram.")
            continue
        bins = hist_row['bins'].iloc[0]
        
        opId = filterDict['opId'][i]
        op = encoding.idx2op.get(opId, 'NA')
        val = filterDict['val'][i]
        
        # Normalize the value based on column min-max
        mini, maxi = encoding.column_min_max_vals.get(col, (0, 1))
        val_unnorm = val * (maxi - mini) + mini
        
        # Determine the bin range based on the operator
        if op == '=':
            # Find the bin where val_unnorm fits
            bin_idx = np.digitize(val_unnorm, bins) - 1
            if 0 <= bin_idx < bins_per_filter:
                res[i, bin_idx] = 1
        elif op == '<':
            bin_idx = np.digitize(val_unnorm, bins) - 1
            if bin_idx > 0:
                res[i, :bin_idx] = 1
        elif op == '>':
            bin_idx = np.digitize(val_unnorm, bins) - 1
            if bin_idx < bins_per_filter - 1:
                res[i, bin_idx + 1:] = 1
        else:
            logging.warning(f"Unsupported operator '{op}'. Using zero histogram for this filter.")
    
    # Flatten the histogram features
    hist_features = res.flatten()
    return hist_features



def formatJoin(json_node):
   
    join = None
    if 'Hash Cond' in json_node:
        join = json_node['Hash Cond']
    elif 'Join Filter' in json_node:
        join = json_node['Join Filter']
    ## TODO: index cond
    elif 'Index Cond' in json_node and not json_node['Index Cond'][-2].isnumeric():
        join = json_node['Index Cond']
    
    ## sometimes no alias, say t.id 
    ## remove repeat (both way are the same)
    if join is not None:

        twoCol = join[1:-1].split(' = ')
        twoCol = [json_node['Alias'] + '.' + col 
                  if len(col.split('.')) == 1 else col for col in twoCol ] 
        join = ' = '.join(sorted(twoCol))
    
    return join
    
def formatFilter(plan):
    alias = None
    if 'Alias' in plan:
        alias = plan['Alias']
    else:
        pl = plan
        while 'parent' in pl:
            pl = pl['parent']
            if 'Alias' in pl:
                alias = pl['Alias']
                break
    
    filters = []
    if 'Filter' in plan:
        filters.append(plan['Filter'])
    if 'Index Cond' in plan and plan['Index Cond'][-2].isnumeric():
        filters.append(plan['Index Cond'])
    if 'Recheck Cond' in plan:
        filters.append(plan['Recheck Cond'])
        
    return filters, alias

class Encoding:
    def __init__(self, column_min_max_vals, 
                 col2idx, op2idx={'>':0, '=':1, '<':2, 'NA':3}):
        
        self.column_min_max_vals = column_min_max_vals
        self.col2idx = col2idx
        self.op2idx = op2idx
        
        idx2col = {}
        for k,v in col2idx.items():
            idx2col[v] = k
        self.idx2col = idx2col
        self.idx2op = {0:'>', 1:'=', 2:'<', 3:'NA'}
        
        self.type2idx = {'Gather': 0, 'Hash Join': 1, 'Seq Scan': 2, 'Hash': 3, 'Bitmap Heap Scan': 4, 'Bitmap Index Scan': 5, 'Nested Loop': 6, 'Index Scan': 7, 'Merge Join': 8, 'Gather Merge': 9, 'Materialize': 10, 'BitmapAnd': 11, 'Sort': 12}
        self.idx2type = {v:k for k,v in self.type2idx.items()}
        self.join2idx = {None: 0, 'mi_idx.movie_id = t.id': 1, 'mc.movie_id = t.id': 2, 'mi.movie_id = t.id': 3, 'ci.movie_id = t.id': 4, 'mk.movie_id = t.id': 5, 'ci.movie_id = mk.movie_id': 6, 'mi.movie_id = mk.movie_id': 7, 'mi_idx.movie_id = mk.movie_id': 8, 'mc.movie_id = mk.movie_id': 9, 'ci.movie_id = mi_idx.movie_id': 10, 'ci.movie_id = mc.movie_id': 11, 'ci.movie_id = mi.movie_id': 12, 'mi.movie_id = mi_idx.movie_id': 13, 'mc.movie_id = mi_idx.movie_id': 14, 'mc.movie_id = mi.movie_id': 15}
        self.idx2join = {v:k for k,v in self.join2idx.items()}
        
        self.table2idx = {'NA': 0, 'title': 1, 'movie_info_idx': 2, 'movie_info': 3, 'movie_companies': 4, 'movie_keyword': 5, 'cast_info': 6}
        self.idx2table = {v:k for k,v in self.table2idx.items()}
    
    def normalize_val(self, column, val, log=False):
        mini, maxi = self.column_min_max_vals[column]
        
        val_norm = 0.0
        if maxi > mini:
            val_norm = (val-mini) / (maxi-mini)
        return val_norm
    
    def encode_filters(self, filters=[], alias=None): 
        ## filters: list of dict 

#        print(filt, alias)
        if len(filters) == 0:
            return {'colId':[self.col2idx['NA']],
                   'opId': [self.op2idx['NA']],
                   'val': [0.0]} 
        res = {'colId':[],'opId': [],'val': []}
        for filt in filters:
            filt = ''.join(c for c in filt if c not in '()')
            fs = filt.split(' AND ')
            for f in fs:
     #           print(filters)
                col, op, num = f.split(' ')
                column = alias + '.' + col
    #            print(f)
                
                res['colId'].append(self.col2idx[column])
                res['opId'].append(self.op2idx[op])
                res['val'].append(self.normalize_val(column, float(num)))
        return res
    
    def encode_join(self, join):
        if join not in self.join2idx:
            self.join2idx[join] = len(self.join2idx)
            self.idx2join[self.join2idx[join]] = join
        return self.join2idx[join]
    
    def encode_table(self, table):
        if table not in self.table2idx:
            self.table2idx[table] = len(self.table2idx)
            self.idx2table[self.table2idx[table]] = table
        return self.table2idx[table]

    def encode_type(self, nodeType):
        if nodeType not in self.type2idx:
            self.type2idx[nodeType] = len(self.type2idx)
            self.idx2type[self.type2idx[nodeType]] = nodeType
        return self.type2idx[nodeType]


class TreeNode:
    def __init__(self, nodeType, typeId, filt, card, join, join_str, filterDict):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt
        
        self.table = 'NA'
        self.table_id = 0
        self.query_id = None ## so that sample bitmap can recognise
        
        self.join = join
        self.join_str = join_str
        self.card = card #'Actual Rows'
        self.children = []
        self.rounds = 0
        
        self.filterDict = filterDict
        
        self.parent = None
        
        self.feature = None
        
    def addChild(self,treeNode):
        self.children.append(treeNode)
    
    def __str__(self):
#        return TreeNode.print_nested(self)
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filter, self.join_str, len(self.children))

    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def print_nested(node, indent = 0): 
        print('--'*indent+ '{} with {} and {}, {} childs'.format(node.nodeType, node.filter, node.join_str, len(node.children)))
        for k in node.children: 
            TreeNode.print_nested(k, indent+1)
        





