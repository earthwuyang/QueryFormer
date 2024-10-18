# model/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque
from .database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist
from .database_util import *

class PlanTreeDataset(Dataset):
    def __init__(self, json_df: pd.DataFrame, train: pd.DataFrame, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample):
        self.table_sample = table_sample  # List of dictionaries per query_id
        self.encoding = encoding
        self.hist_file = hist_file
        self.length = len(json_df)
        
        nodes = [json.loads(plan)['Plan'] for plan in json_df['json']]
        self.cards = [node['Actual Rows'] for node in nodes]
        self.costs = [json.loads(plan)['Execution Time'] for plan in json_df['json']]
        
        self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards)).float()
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs)).float()
        
        self.to_predict = to_predict
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
        elif to_predict == 'card':
            self.gts = self.cards
            self.labels = self.card_labels
        elif to_predict == 'both':
            self.gts = self.costs
            self.labels = (self.cost_labels, self.card_labels)
        else:
            raise Exception('Unknown to_predict type')
        
        idxs = list(json_df['id'])
        self.treeNodes = []
        self.collated_dicts = [self.js_node2dict(i, node) for i, node in zip(idxs, nodes)]
    
    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        return collated_dict
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx])
    
    def pre_collate(self, the_dict, max_node=30, rel_pos_max=20):
        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N+1, N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[edge_index[0, :], edge_index[1, :]] = True
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy(shortest_path_result).long()
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)
        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        
        return {
            'x': x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }
    
    def node2dict(self, treeNode):
        '''
        Converts a tree node into a structured dictionary format
        '''

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(np.array(features)),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
          
        }
    
    def topo_sort(self, root_node):
        '''
        Performs a topological sort on the tree to get the adjacency list and features
        '''
        adj_list = []  # from parent to children
        num_child = []
        features = []

        # Initialize a deque for BFS traversal and add the root node to visit, starting with an index of 0 for the root
        toVisit = deque()
        toVisit.append((0, root_node))
        next_id = 1

        # Loop while there are nodes to visit
        while toVisit:
            idx, node = toVisit.popleft()
            features.append(node.feature)
            num_child.append(len(node.children))

            # Iterate through the children of the current node, add them to the `toVisit` deque, and build the adjacency list
            for child in node.children:
                toVisit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    def traversePlan(self, plan, idx, encoding):
        '''
        Recursively constructs a tree structure from a JSON query plan
        '''

        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = plan.get('Actual Rows', 0)
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)
        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        
        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx
        
        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list, tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        # Separate the parent and child nodes from the adjacency list
        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        # Initialize height counter
        n = 0
        while uneval_nodes.any():
            # Creating a mask for unevaluated child nodes and a mask for unready parents
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            # Determine which nodes can be evaluated
            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1

        return node_order 

def node2feature(node, encoding, hist_file, table_sample):
    '''
    Constructs a feature vector for a given tree node
    '''

    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3, 3-num_filter))
    filts = np.array(list(node.filterDict.values()))  # cols, ops, vals
    filts = np.concatenate((filts, pad), axis=1).flatten() 
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])
    
    hists = filterDict2Hist(hist_file, node.filterDict, encoding)

    # If node.query_id is out of range, return zero sample
    if node.query_id < 0 or node.query_id >= len(table_sample):
        logging.warning(f"query_id {node.query_id} is out of range for table_sample. Using zero sample.")
        sample = np.zeros(1000)
    else:
        if node.table not in table_sample[node.query_id]:
            logging.warning(f"Table '{node.table}' not found in table_sample for query_id={node.query_id}. Using zero sample.")
            sample = np.zeros(1000)
        else:
            sample = table_sample[node.query_id][node.table]
    
    # Ensure sample is of correct length
    if len(sample) != 1000:
        logging.warning(f"Sample length for table '{node.table}' in query_id={node.query_id} is {len(sample)}, expected 1000. Using zero sample.")
        sample = np.zeros(1000)
    
    # Return concatenated feature vector
    return np.concatenate((type_join, filts, mask, hists, [node.table_id], sample))

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def collator(small_set):
    """
    Collates a list of samples into a Batch object.
    """
    batch_dicts, labels = zip(*small_set)
    xs = [s['x'] for s in batch_dicts]
    attn_bias = [s['attn_bias'] for s in batch_dicts]
    rel_pos = [s['rel_pos'] for s in batch_dicts]
    heights = [s['heights'] for s in batch_dicts]
    
    # Concatenate tensors
    x = torch.cat(xs, dim=0)
    attn_bias = torch.cat(attn_bias, dim=0)
    rel_pos = torch.cat(rel_pos, dim=0)
    heights = torch.cat(heights, dim=0)
    
    y = torch.stack(labels, dim=0)
    
    return Batch(attn_bias, rel_pos, heights, x), y

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
    
    def __len__(self):
        return self.x.size(0)
