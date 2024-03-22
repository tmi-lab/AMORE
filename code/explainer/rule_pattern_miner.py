# -----------------------------------------------------------------------------------------
# This work is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
#
# Author: Yu Chen
# Year: 2023
# Description: This file contains the implementation of the regional rule extraction method AMORE.
# -----------------------------------------------------------------------------------------



import numpy as np
import torch
import operator
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer


op_map = {'>=':operator.ge,'>':operator.gt,'<=':operator.le,'<':operator.lt,'==':operator.eq}


from .itemsets_miner import *
from .RuleGrowth_tree import RuleTree,RuleNode


def scan_feature_cond_prob_ratio(f_val,z_indices,grids,prev_cond_indices=None):
    ratios = np.zeros(len(grids)-1)
    supports = np.zeros(len(grids)-1)
    for i in range(len(grids)-1):
        left = grids[i]
        right = grids[i+1]
        rt,sup = calc_cond_ratio(f_val,left,right,z_indices,prev_cond_indices)
        supports[i] = sup
        ratios[i] = rt

    return ratios,supports


def scan_cat_feature_cond_prob_ratio(f_val,z_indices,prev_cond_indices=None):
    cats = np.unique(f_val)
    ratios = np.zeros(len(cats))
    supports = np.zeros(len(ratios))
    for i in range(len(ratios)):
        left = cats[i]
        right = cats[i]
        rt,sup = calc_cond_ratio(f_val,left,right,z_indices,prev_cond_indices)
        supports[i] = sup
        ratios[i] = rt

    return ratios,supports



def calc_cond_ratio(f_val,left_val,right_val,z_indices,prev_cond_indices=None):
    if prev_cond_indices is not None:
        cond_indices = z_indices & prev_cond_indices
    else:
        cond_indices = z_indices
        prev_cond_indices = np.ones_like(f_val).astype(bool)

    support = np.sum((f_val[prev_cond_indices]>=left_val)&(f_val[prev_cond_indices]<=right_val))
    if support == 0:
        ratio = -1.
        return ratio,support
    
    dom = np.sum(cond_indices)/np.sum(prev_cond_indices)
    ratio = (np.sum((f_val[cond_indices]>=left_val)&(f_val[cond_indices]<=right_val))/support)/dom

    return ratio,support





def remove_duplicate_rules(rules_dict):
    keys = list(rules_dict.keys())
    cnt = 0
    for i in range(len(rules_dict)-1):
        for j in range(i+1,len(rules_dict)):
            ### need to consider different order with same rules 
            if set(keys[i]) == set(keys[j]) and len(rules_dict[keys[i]]["rules"]) == len(rules_dict[keys[j]]["rules"]):
                duplicate = True
                #print("check duplicate rules keys",keys[i],keys[j])
                for ii, pi in enumerate(keys[i]):
                    jj = list(keys[j]).index(pi)
                    for k in range(2):
                        ri = rules_dict[keys[i]]["rules"][ii*2+k]
                        rj = rules_dict[keys[j]]["rules"][jj*2+k]
                        #print("check duplicate rules",ii,ri,jj,rj)
                        if ri[0] != rj[0] or ri[1] != rj[1] or ri[2] != rj[2]:
                            duplicate = False
                            break
                if duplicate:  
                    print("remove duplicate rules",keys[i],rules_dict[keys[i]]["rules"],keys[j],rules_dict[keys[j]]["rules"])                  
                    del rules_dict[keys[j]]
                    return remove_duplicate_rules(rules_dict)  
                        
    return rules_dict





def search_feature_intervals(f_val,peaks,grids,ratios,supports,target_indices,prev_cond_indices=None,min_support=2000,top_K=1,local=False,verbose=False):
    intervals = []
    for gid in peaks:        
        cond_ratio,left,right,sup = raise_feature_interval(f_val,grids,gid,ratios=ratios,supports=supports,target_indices=target_indices,
                                                           prev_cond_indices=prev_cond_indices,min_support=min_support,local=local,verbose=verbose)     

        if sup < min_support or cond_ratio < 1.0001:
            #print('not qualified',sup,cond_ratio)
            continue
        else:          
            valid = True
            if left == f_val.min() and right == f_val.max():
                valid = False
                
            if valid:
                if prev_cond_indices is None:
                    new_prev_cond_indices = (f_val>=left)&(f_val<=right)
                else:
                    new_prev_cond_indices = prev_cond_indices & ((f_val>=left)&(f_val<=right))
                intervals.append((cond_ratio,left,right,sup,new_prev_cond_indices))
                if len(intervals) >= top_K:
                    break
    
    if len(intervals) == 0:
        return intervals
    
    intervals.sort(key=lambda x: x[0], reverse=True) 
    return intervals   



def add_potential_rules_for_categorical_feature(x,f,target_indices,prev_cond_indices=None,min_support=2000,
                        local_x=None,top_K=1,verbose=False):
    cats = np.unique(x[:,int(f)])
    ratios,supports = scan_cat_feature_cond_prob_ratio(x[:,int(f)],target_indices,prev_cond_indices=prev_cond_indices)
    lx = None
    if local_x is not None:
        if isinstance(local_x,torch.Tensor):
            local_x = local_x.numpy()
        lx = local_x[int(f)]

    inv = []
    for i,(r,s) in enumerate(zip(ratios,supports)):
        if s >= min_support and r > 1.0001:
            if lx is not None and lx != cats[i]:
                    continue
            else:
                if prev_cond_indices is None:
                    new_prev_cond_indices = (x[:,int(f)]==cats[i])
                else:
                    new_prev_cond_indices = prev_cond_indices & (x[:,int(f)]==cats[i])
                inv.append((r,cats[i],cats[i],s,new_prev_cond_indices))
    if len(inv) == 0:
        return inv
    inv.sort(key=lambda x: x[0], reverse=True)
    return inv[:top_K]
    


def add_potential_rules_for_numerical_feature(x,f,target_indices,prev_cond_indices=None,num_grids=20,min_support=2000,
                        local_x=None,top_K=1,bin_strategy="kmeans",verbose=False):
    f_val = x[:,int(f)]
    est = KBinsDiscretizer(n_bins=num_grids, encode='ordinal', strategy=bin_strategy, subsample=None)
    est.fit(f_val.reshape(-1,1))
    grids = est.bin_edges_[0]
    #grids = np.linspace(f_val[~np.isnan(f_val)].min(),f_val[~np.isnan(f_val)].max(),num_grids+1) 

    grids,ratios,supports = preprocess_empty_grids(f_val,target_indices,grids,prev_cond_indices=prev_cond_indices,verbose=verbose)
    
    lx = None
    if local_x is not None:
        if isinstance(local_x,torch.Tensor):
            local_x = local_x.numpy()
        lx = local_x[int(f)]
  

    peaks = [] if len(ratios)<2 else find_peaks(ratios,verbose=verbose)
        
    inv = search_feature_intervals(f_val,peaks,grids,ratios,supports,target_indices,prev_cond_indices=prev_cond_indices,
                                    min_support=min_support,top_K=top_K,verbose=verbose)

    if lx is not None:
        ## check if local_x is in the candidate range
        for i in range(len(inv)-1,-1,-1):
            if lx < inv[i][1] or lx > inv[i][2]:
                if verbose:
                    print("local not matched",lx,inv[i][:-1])
                del inv[i]
        if len(inv) == 0:
            if verbose: 
                print("no matched interval, search from local val grid")
            tmp = np.arange(len(grids)-1)[(grids[:-1] -lx)<=0]
            if len(tmp)>0:
                gid = tmp[-1]
                inv = search_feature_intervals(f_val,[gid],grids,ratios,supports,target_indices,prev_cond_indices=prev_cond_indices,
                                    min_support=min_support,top_K=top_K,local=True,verbose=verbose)
           
    return inv
        



def add_potential_rules(x,f,target_indices,prev_cond_indices=None,num_grids=20,min_support=2000,
                        local_x=None,top_K=1,feature_type="float",bin_strategy="kmeans",verbose=False):
    if verbose:
        print("search rule for feature",f)   
    if feature_type != "cat":
        return add_potential_rules_for_numerical_feature(x,f,target_indices,prev_cond_indices=prev_cond_indices,num_grids=num_grids,
                                                        min_support=min_support,local_x=local_x,top_K=top_K,bin_strategy=bin_strategy,verbose=verbose)
    else:
        return add_potential_rules_for_categorical_feature(x,f,target_indices,prev_cond_indices=prev_cond_indices,
                                                        min_support=min_support,local_x=local_x,top_K=top_K,verbose=verbose)


def preprocess_empty_grids(f_val,target_indices,grids,prev_cond_indices=None,verbose=False):
    ratios,supports = scan_feature_cond_prob_ratio(f_val,target_indices,grids,prev_cond_indices=prev_cond_indices)
    if verbose:
        print("initial check grids",len(grids),grids)
        print("initial check ratios",len(ratios),ratios)
        print("initial check supports",len(supports),supports)
    grids = merge_consecutive_equal_grids(grids,ratios)
    ratios,supports = scan_feature_cond_prob_ratio(f_val,target_indices,grids,prev_cond_indices=prev_cond_indices)
    if verbose:
        print("merge 1 check grids",len(grids),grids)
        print("merge 1 check ratios",len(ratios),ratios)
        print("merge 1 check supports",len(supports),supports)
    grids = merge_empty_grids(grids,ratios,supports)   
    ratios,supports = scan_feature_cond_prob_ratio(f_val,target_indices,grids,prev_cond_indices=prev_cond_indices)
    if verbose:
        print("merge 2 check grids",len(grids),grids)
        print("merge 2 check ratios",len(ratios),ratios)
        print("merge 2 check supports",len(supports),supports)
    grids = merge_consecutive_equal_grids(grids,ratios)
    ratios,supports = scan_feature_cond_prob_ratio(f_val,target_indices,grids,prev_cond_indices=prev_cond_indices)
    if verbose:
        print("merge 3 check grids",len(grids),grids)
        print("merge 3 check ratios",len(ratios),ratios)
        print("merge 3 check supports",len(supports),supports)
    return grids,ratios,supports
 


def merge_empty_grids(grids,ratios,supports):    
    sids = supports==0
    for s in np.arange(len(supports))[sids]:
        merge = s
        l = s-1
        r = s+1
        if l >= 0:
            r_l = ratios[l]
        else:
            merge = r
        if r < len(ratios):
            r_r = ratios[r]
        else:
            merge = l 
        if merge == s:
            merge = l if r_l >= r_r else r
        grids[s] = grids[merge]
        ratios[s] = ratios[merge]
    new_grids = np.unique(grids)
    return new_grids


def merge_consecutive_equal_grids(grids,ratios):
    eids = np.arange(len(ratios)-1)[np.diff(ratios)==0]

    for s in eids:
        grids[s+1] = grids[s]

    new_grids = np.unique(grids)
    return new_grids


def find_peaks(ratios,verbose=False):
    peaks = []
    for i in range(len(ratios)):
        if ratios[i]<=1.:
            continue
        if i==0 and ratios[i]>ratios[i+1]:
            peaks.append(i)
        elif i==len(ratios)-1 and ratios[i]>ratios[i-1]:
            peaks.append(i)
        elif ratios[i]>ratios[i-1] and ratios[i]>ratios[i+1]:
            peaks.append(i)
            
    if len(peaks)>0:
        peaks = np.array(peaks)
        if verbose:
            print("check peaks",peaks,ratios)
        peaks = peaks[np.argsort(ratios[peaks])[::-1]]
        if verbose:
            print("check sorted peaks",ratios[peaks],ratios)
    return peaks


def raise_feature_interval(f_val,grids,gid,ratios,supports,target_indices,prev_cond_indices=None,min_support=2000,local=False,verbose=False):

    if prev_cond_indices is None:
        sup = supports[gid]
    else:
        sup = np.sum((f_val[prev_cond_indices]>=grids[gid])&(f_val[prev_cond_indices]<=grids[gid+1]))

    ratio,left_id,right_id,sup = merge_feature_intervals(gid,sup,f_val,grids,ratios,target_indices,
                                                       prev_cond_indices=prev_cond_indices,
                                                       min_support=min_support,local=local,verbose=verbose)
         
    ### check the support of the new range, if local_x is not None, check if local_x is in the new range
    if sup < min_support:
            return -1.,-1,-1,-1
    if verbose:
        print("check raise",gid,sup,ratio,grids[left_id],grids[right_id])    
    return ratio,grids[left_id],grids[right_id],sup


def merge_feature_intervals(gid,sup,f_val,grids,ratios,target_indices,
                            prev_cond_indices=None,min_support=2000,
                            local=False,verbose=False):
    
    left_id = gid
    right_id = gid
    old_r = ratios[gid]
    r_limit = 1.0001
    if verbose:
        print("merge_feature_intervals",gid,sup,old_r,grids[gid])
    while old_r > r_limit or sup < min_support:
        
        # if left_id == 0 or right_id == len(ratios)-1:
        #     break
        
        new_left_id = left_id - 1
        new_right_id = right_id + 1
        if verbose:
            print("check before merge ids: left {}, right {}, new left {}, new right{}".format(left_id,right_id,new_left_id,new_right_id))

        left_r = ratios[new_left_id] if new_left_id >= 0 else -1.
        right_r = ratios[new_right_id] if new_right_id < len(ratios) else -1.
        
        ## merge neighbor grid with ratio larger than 1
        if left_r<0 and right_r<0:
            break
        if left_r >= right_r or new_right_id >= len(ratios):
            merge = new_left_id
            m_r = left_r
        else:
            merge = new_right_id
            m_r = right_r
        if verbose:
            print("check merge ids",left_id,right_id,merge,m_r,old_r,left_r,right_r)    
        if sup >= min_support and (m_r < old_r or local):
            # left_id = left_id + 1
            # right_id = right_id - 1
            break
        
        # left_id = max(0,left_id)
        # right_id = min(len(ratios)-1,right_id)
        
        new_r,new_sup = calc_cond_ratio(f_val,grids[min(merge,left_id)],grids[max(merge,right_id)+1],
                                        target_indices,prev_cond_indices)

        old_r = new_r
        left_id = min(merge,left_id)
        right_id = max(merge,right_id)
        sup = new_sup
    if verbose:
        print("check after merge",left_id,right_id,sup,old_r)
    return old_r,left_id,right_id+1,sup


def display_rules(rules,x,target_indices,y=None,c=-1,verbose=False,ftypes=None):
    #print("display",c)
    if not isinstance(rules,list):
        rules = [rules]
    for r in rules:
        
        #r =rules[k]
        f = int(r[0]) 
        ## remove useless rules
        if op_map[r[-2]](x[...,f],r[-1]).sum() == len(x):
            rules.remove(r)
        # elif ftypes is not None and ftypes[f] == 'int':
        #     rv = np.floor(r[2]) if r[1] == '>=' else np.ceil(r[2])
        #     rules[k] = (r[0],r[1],rv)
        #     print("rule int",rules[k])
    h_cond_prob_z,h_cond_prob_y, h_ratio_y,h_sup = target_prob_with_rules(rules,x,zids=target_indices,y=y,c=c,verbose=verbose) 
    fitness = (2.*h_cond_prob_z - 1.)*h_sup/np.sum(target_indices)
    
    ret = {"rules":rules,
            "confidence":h_cond_prob_z,
            "support":h_sup,
            "fitness":fitness}
    if y is not None:
        ret["cond_prob_y"] = h_cond_prob_y
        ret["ratio_y"] = h_ratio_y
    return ret

def confine_int_feature_rules(rules):

    if not isinstance(rules,list):
        rules = [rules]
    for k in range(len(rules)):
        r =rules[k]
        rv = np.ceil(r[2]) if r[1] == '>=' else np.floor(r[2])
        rules[k] = (r[0],r[1],rv)

    if len(rules) == 2 and (rules[0][2] == rules[1][2]):        
        rules = [(rules[0][0],"==",rules[0][2])]
    return rules
    


def target_prob_with_rules(rule_list,x,zids=None,y=None,c=-1,verbose=False):
    mask = np.ones_like(y).astype(bool)
    for r in rule_list:
        mask = mask & op_map[r[-2]](x[...,int(r[0])],r[-1])
        if verbose:
            print(r,np.sum(mask))
    if y is None:
        return np.sum(mask & zids)/np.sum(mask),np.nan,np.nan,np.sum(mask)
    if zids is None:
        return np.nan,np.sum(y[mask]==c)/np.sum(mask),np.sum(y[mask]==c)/np.sum(y==c),np.sum(mask)
    
    return np.sum(mask & zids)/np.sum(mask), np.sum(y[mask]==c)/np.sum(mask),np.sum(y[mask]==c)/np.sum(y==c),np.sum(mask)




def gen_rule_list_for_one_target(x,fids,target_indices,y=None,c=1,min_support=500,num_grids=20,max_depth=5,top_K=3,
                                        local_x=None,feature_types=None,search="greedy",bin_strategy="kmeans",
                                        verbose=False,sort_by="fitness"):
    
    rule_tree = build_rule_tree(list(fids),x,target_indices,grid_num=num_grids,min_support=min_support,
                                max_depth=max_depth,top_K=top_K,local_x=local_x,search=search,
                                feature_types=feature_types,bin_strategy=bin_strategy,verbose=verbose)
    if rule_tree is None:
        rule_dict = {}
    else:
        _,rule_dict = rule_tree.get_rule_dict()
    
    rule_list = []
    for rules in rule_dict.values():
        rlist = rules["rules"]
        assert len(rlist)%2 == 0
        new_rlist = []
        for k in range(0,len(rlist),2):
            f = int(rlist[k][0])
            r = rlist[k:k+2]
            if feature_types is not None:
                if 'int' in feature_types[int(f)]:                
                    r = confine_int_feature_rules(r)
                elif 'cat' in feature_types[int(f)]:
                    r = [(r[0][0],"==",r[0][2])]
            new_rlist += r

        rules["rules"] = new_rlist
        rule_list.append(display_rules(rules["rules"],x,target_indices,y,c=c,verbose=verbose))
    if len(rule_list) > 1:
        rule_list.sort(key=lambda x: x[sort_by], reverse=True)
    return rule_list



def build_rule_tree(fids,x,target_indices,grid_num=20,min_support=2000,max_depth=4,top_K=3,local_x=None,search="ordered",bin_strategy="kmeans",feature_types=None,verbose=False):
    # print("build_rule_tree")
    rule_tree = RuleTree(min_support=min_support)
    add_branch_to_rule_tree(rule_tree.root,fids,x,target_indices,prev_cond_indices=None,path=[],grid_num=grid_num,
                        min_support=min_support,max_depth=max_depth,top_K=top_K,local_x=local_x,search=search,
                        feature_types=feature_types,bin_strategy=bin_strategy,verbose=verbose)
    return rule_tree



def add_branch_to_rule_tree(parent,fids,x,target_indices,prev_cond_indices=None,path=[],grid_num=20,
                            min_support=2000,max_depth=4,top_K=3,local_x=None,search="greedy",
                            feature_types=None,bin_strategy="kmeans",verbose=False):
    fids_copy = fids.copy()
    r_limit = 1.0001
    if parent.fid != -1:
        fids_copy.remove(parent.fid)
    if len(fids_copy)>0:
        if search == "ordered":
            f = fids_copy[0]
            feature_type="float" if feature_types is None else feature_types[f]
            potential_rules = add_potential_rules(x,f,target_indices,prev_cond_indices=prev_cond_indices,num_grids=grid_num,
                                                min_support=min_support,top_K=top_K,local_x=local_x,
                                                feature_type=feature_type,bin_strategy=bin_strategy,verbose=verbose)
           
            best_potential_rules = [[potential_rules[k][0],f, potential_rules[k]] for k in range(len(potential_rules))]
            
            
        if search == "greedy":
            # initialize top_K best rules with the lower ratio limit   
            best_potential_rules = [[r_limit,fids_copy[0],None]]*top_K
            for f in fids_copy:
                feature_type="float" if feature_types is None else feature_types[f]
                potential_rules = add_potential_rules(x,f,target_indices,prev_cond_indices=prev_cond_indices,num_grids=grid_num,
                                                min_support=min_support,top_K=top_K,local_x=local_x,
                                                feature_type=feature_type,bin_strategy=bin_strategy,verbose=verbose) 
                if len(potential_rules) == 0:
                    continue
                for k in range(len(potential_rules)):
                    
                    if potential_rules[k][0] <= best_potential_rules[-1][0]:
                        break
                    # replace the worst rule with the new rule if the new rule is better
                    if potential_rules[k][0] > best_potential_rules[-1][0]:
                        best_potential_rules[-1] = [potential_rules[k][0],f,potential_rules[k]]
                        # sort the best rules again to keep the worst one at the end
                        best_potential_rules.sort(key=lambda x: x[0], reverse=True)
                        break
            if verbose:            
                print('best rule',best_potential_rules)
            
            if best_potential_rules[0][0]<=r_limit:
                # nothing to add, just return
                return
            
        valid = False     
        for k, potential_rule in enumerate(best_potential_rules):
            if potential_rule[0] == r_limit:
                break
            f = potential_rule[1]
            cond_ratio, left, right, sup, new_prev_cond_indices = potential_rule[-1]
            if verbose:
                print('check potential rule',f,cond_ratio,left,right,sup)
            if sup < min_support or cond_ratio <= r_limit:
                if verbose:
                    print('no enough support or ratio,skip',sup,cond_ratio)
                
            else:  
                if verbose:        
                    print('add rule',path, f, potential_rule[-1][:-1])  
                valid = True                  
                new_node = RuleNode(f,parent,left,right,cond_ratio,sup)
                parent.add_child(new_node)
                path_copy = path.copy()
                path_copy.append(f)
                if len(path_copy) < max_depth:
                    add_branch_to_rule_tree(new_node,fids_copy,x,target_indices,prev_cond_indices=new_prev_cond_indices,
                                            path=path_copy,grid_num=grid_num,min_support=min_support,max_depth=max_depth,
                                            top_K=top_K,local_x=local_x,search=search,feature_types=feature_types,bin_strategy=bin_strategy,verbose=verbose)
        if not valid and search == "ordered":
            if verbose:
                print('no valid rule,skip',f)
            fids_copy.remove(f)
            if parent.fid!=-1:
                fids_copy.insert(0,parent.fid)
            add_branch_to_rule_tree(parent,fids_copy,x,target_indices,prev_cond_indices=prev_cond_indices,path=path,grid_num=grid_num,
                                    min_support=min_support,max_depth=max_depth,top_K=top_K,local_x=local_x,search=search,
                                    feature_types=feature_types,bin_strategy=bin_strategy,verbose=verbose)

    return




def replace_feature_names(rules,input_feature_names,time_index=False):
    new_r = []
    if time_index is False:
        for r in rules:
            new_r.append((r[0],input_feature_names[r[0]],r[1],r[2]))
    else:
        f_len = len(input_feature_names)
        for r in rules:
            new_r.append((r[0],input_feature_names[int(r[0]%f_len)]+"_t"+str(int(r[0]/f_len)),r[1],r[2]))
    return new_r

 
    
def param_grid_search_for_amore(bin_strategies,ng_range,support_range,X,fids,target_indices,y,c=1,confidence_lower_bound = 0.75,
                                max_depth=1,top_K=3,sort_by="fitness",feature_types=None,local_x=None,verbose=False):
    best_rule_set = None
    best_fitness,best_confidence = 0., 0.
    best_configs = None
    config_metric_records = {}
    print("grid search hyperparameters")
    for bin_strategy in bin_strategies:
        for num_grids in ng_range:   
            config_metric_records[(bin_strategy,num_grids)]={"min_supports":support_range}
            top_confidence_records,top_fitness_records, actual_supports = [],[],[]
            for min_support in support_range:
                # print("check config",bin_strategy,num_grids,min_support)
                y_rule_candidates = gen_rule_list_for_one_target(X,fids,target_indices,y=y,c=c,sort_by=sort_by,
                                                        min_support=min_support,num_grids=num_grids,max_depth=max_depth,top_K=top_K,
                                                        local_x=local_x,feature_types=feature_types,bin_strategy=bin_strategy,
                                                        verbose=verbose)
                if len(y_rule_candidates) == 0:
                    top_confidence_records.append(0)
                    top_fitness_records.append(0)            
                    actual_supports.append(0)
                    continue
                top_fitness = y_rule_candidates[0]["fitness"]
                top_confidence = y_rule_candidates[0]["confidence"]
                # print("check top",y_rule_candidates[0])
                top_confidence_records.append(top_confidence)
                top_fitness_records.append(top_fitness)            
                actual_supports.append(y_rule_candidates[0]["support"])
                if top_confidence >= confidence_lower_bound:
                    if sort_by == "fitness" and top_fitness > best_fitness:
                        best_rule_set = y_rule_candidates[0]
                        best_fitness = top_fitness
                        best_configs = {"bin_strategy":bin_strategy, "num_grids":num_grids, "min_support":min_support}

                    elif sort_by == "confidence" and top_confidence > best_confidence:
                        best_rule_set = y_rule_candidates[0]
                        best_confidence = top_confidence
                        best_configs = {"bin_strategy":bin_strategy, "num_grids":num_grids, "min_support":min_support}
            config_metric_records[(bin_strategy,num_grids)]["top_confidence_records"]=top_confidence_records
            config_metric_records[(bin_strategy,num_grids)]["top_fitness_records"]=top_fitness_records
            config_metric_records[(bin_strategy,num_grids)]["actual_support"]=actual_supports
    return best_rule_set, best_configs, config_metric_records




