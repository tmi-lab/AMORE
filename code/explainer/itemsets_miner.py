
import numpy as np
from .FPGrowth_tree import FPTree


### get indices of raw features that has int_j larger than threshold
### valid indices start from 1, 0 indicates features that are lower than threshold
def masked_indices(int_j,D,threshold=0.1):
    """ 
    D: number of raw features
    """
    d = len(int_j.shape)-1
    indices = np.expand_dims(np.arange(1,D+1),axis=[i for i in range(d)])*np.ones_like(int_j)
    mask = np.zeros_like(int_j)
    mask[int_j>=threshold]=1.
    indices = indices*mask
    return indices


### generate itemsets from masked indices
def gen_itemsets(masked_indices,K=48):
    """
    masked_indices: numpy array of shape (N,K,D)
    K: number of latent features
    """
    itemsets = {}
    for z in range(K):
        itemsets[z]=[]
        for n,idx in enumerate(masked_indices):
            idx = idx[z]
            iset = list(idx[idx>0])
            if len(iset)>0:
                itemsets[z].append(iset) 
    return itemsets


def transform_intgrad_to_itemsets(int_j,thd=0.1,K=48):
    num_features = int_j.shape[-1]
    
    indices = np.arange(1,num_features+1)#(np.arange(1,num_features+1).reshape(1,1,1,-1)*np.ones_like(int_j))
    for s in range(len(int_j.shape)-1):
        indices = np.expand_dims(indices,axis=0)
    indices = indices*np.ones_like(int_j)
    #print(indices.shape,indices[:2])
    mask = np.zeros_like(int_j)
    mask[np.abs(int_j)>=thd]=1.
    indices = indices*mask
    indices = indices.reshape(indices.shape[0],-1,indices.shape[-1])
    
    itemsets = gen_itemsets(indices,K=K)
    return itemsets




def gen_freq_itemsets(itemsets_z, min_support=500,max_len=10):
    # print("generating frequent itemsets",max_len)
    fp_tree = FPTree(itemsets_z,min_support)
    freq_itemsets = fp_tree.get_itemsets(min_support,max_depth=max_len)
    return freq_itemsets


def select_feature_combination(freq_itemsets,max_len=5):
    max_cnt = 0
    comb = []
    comb_len = 0
    for s, dc in freq_itemsets.items():
        #print(s,dc['cnt'],len(dc['sids']))
        if len(s)>max_len:
            continue
        if len(s) > comb_len or (dc['cnt']>max_cnt and len(s)==comb_len):
            max_cnt = dc['cnt']
            comb = s
            comb_len = len(s)
    return comb


def gen_freq_feature_set(itemsets,min_support=500,max_len=5):
    freq_itemsets = gen_freq_itemsets(itemsets,min_support=min_support,max_len=max_len)
    freq = select_feature_combination(freq_itemsets,max_len=max_len)
    
    return freq