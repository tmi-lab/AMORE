import numpy as np
import torch
import operator


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


def gen_top_rule_dict_for_one_target(x,fids,target_indices,min_support=500,num_grids=20,max_depth=5,local_x=None,verbose=False):
    
    fids_copy = list(fids).copy()
    prev_cond_indices = None
    rule_dict = {}

    for f in fids_copy:
        ratio,left,right,sup,prev_cond_indices = add_top_rule(x,f,target_indices,prev_cond_indices=prev_cond_indices,num_grids=num_grids,
                        min_support=min_support,max_depth=max_depth,local_x=local_x,verbose=verbose)
        if ratio <= 1.:
            continue
        rule_dict[f] = {"rule":[(f,">=",left),(f,"<=",right)],"support":(ratio,sup)}
        print('add rule',rule_dict[f])
    
    return rule_dict


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


def add_top_rule(x,f,target_indices,prev_cond_indices=None,num_grids=20,
                            min_support=2000,max_depth=4,local_x=None,verbose=False):        
    op_ratio = 0.
    op_left = 0.
    op_right = 0.
    op_sup = 0.
    new_prev_cond_indices = prev_cond_indices
    
    inv = add_potential_rules(x,f,target_indices,prev_cond_indices=prev_cond_indices,num_grids=num_grids,min_support=min_support,
                                local_x=local_x,top_K=1,verbose=verbose)
    if inv is not None:
        op_ratio,op_left,op_right,op_sup,new_prev_cond_indices = inv[0]
                   
    # if local_x is not None:
    #     print("check local",lx,op_left,op_right)
                
    return op_ratio,op_left,op_right,op_sup, new_prev_cond_indices



def search_feature_intervals(f_val,peaks,grids,ratios,supports,target_indices,prev_cond_indices=None,min_support=2000,top_K=1,verbose=False):
    intervals = []
    for gid in peaks:        
        cond_ratio,left,right,sup = raise_feature_interval(f_val,grids,gid,ratios=ratios,supports=supports,target_indices=target_indices,
                                                           prev_cond_indices=prev_cond_indices,min_support=min_support,verbose=verbose)     

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



def add_potential_rules(x,f,target_indices,prev_cond_indices=None,num_grids=20,min_support=2000,
                        local_x=None,top_K=1,verbose=False):
    print("search rule for feature",f)
    f_val = x[:,int(f)]
    grids = np.linspace(f_val[~np.isnan(f_val)].min(),f_val[~np.isnan(f_val)].max(),num_grids) 

    grids,ratios,supports = preprocess_empty_grids(f_val,target_indices,grids,prev_cond_indices=prev_cond_indices,verbose=verbose)

    lx = None
    if local_x is not None:
        if isinstance(local_x,torch.Tensor):
            local_x = local_x.numpy()
        lx = local_x[int(f)]
  

    peaks = find_peaks(ratios,verbose=verbose)
        
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
            print("no matched interval, search from local val grid")
            tmp = np.arange(len(grids)-1)[(grids[:-1] -lx)<=0]
            if len(tmp)>0:
                gid = tmp[-1]
                inv = search_feature_intervals(f_val,[gid],grids,ratios,supports,target_indices,prev_cond_indices=prev_cond_indices,
                                    min_support=min_support,top_K=top_K,verbose=verbose)
        
                
    
    return inv
        
    


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


def raise_feature_interval(f_val,grids,gid,ratios,supports,target_indices,prev_cond_indices=None,min_support=2000,verbose=False):

    if prev_cond_indices is None:
        sup = supports[gid]
    else:
        sup = np.sum((f_val[prev_cond_indices]>=grids[gid])&(f_val[prev_cond_indices]<=grids[gid+1]))

    ratio,left_id,right_id,sup = merge_feature_intervals(gid,sup,f_val,grids,ratios,target_indices,
                                                       prev_cond_indices=prev_cond_indices,
                                                       min_support=min_support,verbose=verbose)
         
    ### check the support of the new range, if local_x is not None, check if local_x is in the new range
    if sup < min_support:
            return -1.,-1,-1,-1
    if verbose:
        print("check raise",gid,sup,ratio,grids[left_id],grids[right_id])    
    return ratio,grids[left_id],grids[right_id],sup


def merge_feature_intervals(gid,sup,f_val,grids,ratios,target_indices,prev_cond_indices=None,min_support=2000,verbose=False):
    
    left_id = gid
    right_id = gid
    old_r = ratios[gid]
    if verbose:
        print("merge_feature_intervals",gid,sup,old_r,grids[gid])
    while old_r > 1.0001 or sup < min_support:
        
        # if left_id == 0 or right_id == len(ratios)-1:
        #     break
        
        new_left_id = left_id - 1
        new_right_id = right_id + 1
        if verbose:
            print("check before merge ids",left_id,right_id,new_left_id,new_right_id) 

        left_r = ratios[new_left_id] if new_left_id >= 0 else -1.
        right_r = ratios[new_right_id] if new_right_id < len(ratios) else -1.
        
        ## merge neighbor grid with ratio larger than 1
        if left_r >= right_r or new_right_id >= len(ratios):
            merge = new_left_id
            m_r = left_r
        else:
            merge = new_right_id
            m_r = right_r
        if verbose:
            print("check merge ids",left_id,right_id,merge,m_r,old_r,left_r,right_r)    
        if sup >= min_support and m_r < old_r:
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
    ret = {"rules":rules,
            "cond_prob_target":h_cond_prob_z,
            "support":h_sup}
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



def find_top_pattern_for_one_target(x,y,target_indices,itemsets,c=1,num_grids=20,omega=0.1,min_support=500,
                                max_depth=4,local_x=None,verbose=False,feature_types=None):

    fids = gen_freq_feature_set(itemsets,min_support=min_support,max_len=max_depth*2)
    fids = np.array(fids).astype(int)-1
    print('feature set',fids)
    rule_dict = gen_top_rule_dict_for_one_target(x,fids,target_indices,min_support=min_support,num_grids=num_grids,
                                             max_depth=max_depth,local_x=local_x,verbose=verbose)


    for p,rules in rule_dict.items():
        print(p,rules)
        if feature_types is not None and feature_types[int(p)] == 'int':
            rule_dict[p]["rule"] = confine_int_feature_rules(rules["rule"])

    processed_rules = {"rules":[rv for r in rule_dict.values() for rv in r["rule"] ]}
    print("check processed rules",processed_rules)
    processed_rules = display_rules(processed_rules["rules"],x,target_indices,y,c=c,verbose=verbose)

        
    return processed_rules


def gen_rule_list_for_one_target(x,fids,target_indices,y=None,c=1,min_support=500,num_grids=20,max_depth=5,top_K=3,
                                        local_x=None,feature_types=None,search="ordered",verbose=False,sort_by="cond_prob_target"):
    
    rule_tree = build_rule_tree(list(fids),x,target_indices,grid_num=num_grids,min_support=min_support,
                                max_depth=max_depth,top_K=top_K,local_x=local_x,search=search,verbose=verbose)
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
            if feature_types is not None and feature_types[int(f)] == 'int':                
                r = confine_int_feature_rules(r)
            new_rlist += r

        rules["rules"] = new_rlist
        rule_list.append(display_rules(rules["rules"],x,target_indices,y,c=c,verbose=verbose))
    if len(rule_list) > 1:
        rule_list.sort(key=lambda x: x[sort_by], reverse=True)
    return rule_list



def build_rule_tree(fids,x,target_indices,grid_num=20,min_support=2000,max_depth=4,top_K=3,local_x=None,search="ordered",verbose=False):
    print("build_rule_tree")
    rule_tree = RuleTree(min_support=min_support)
    add_branch_to_rule_tree(rule_tree.root,fids,x,target_indices,prev_cond_indices=None,path=[],grid_num=grid_num,
                        min_support=min_support,max_depth=max_depth,top_K=top_K,local_x=local_x,search=search,verbose=verbose)
    return rule_tree




def add_branch_to_rule_tree(parent,fids,x,target_indices,prev_cond_indices=None,path=[],grid_num=20,
                            min_support=2000,max_depth=4,top_K=3,local_x=None,search="ordered",verbose=False):
    fids_copy = fids.copy()
    if parent.fid != -1:
        fids_copy.remove(parent.fid)
    if len(fids_copy)>0:
        if search == "ordered":
            f = fids_copy[0]
            potential_rules = add_potential_rules(x,f,target_indices,prev_cond_indices=prev_cond_indices,num_grids=grid_num,
                                                min_support=min_support,top_K=top_K,local_x=local_x,verbose=verbose)
        if search == "greedy":
            best_f = fids_copy[0]
            best_potential_rules = []
            best_r = 0.
            for f in fids_copy:
                # f = fids_copy[0]
                potential_rules = add_potential_rules(x,f,target_indices,prev_cond_indices=prev_cond_indices,num_grids=grid_num,
                                                min_support=min_support,top_K=top_K,local_x=local_x,verbose=verbose) 
                if len(potential_rules) == 0:
                    continue
                if best_r < potential_rules[0][0]:
                    best_r = potential_rules[0][0]
                    best_f = f
                    best_potential_rules = potential_rules   
                        
            print('best rule',best_f,best_r)
            f = best_f
            potential_rules = best_potential_rules
            
            if best_r<=1.0001:
                return
            
        valid = False     
        for potential_rule in potential_rules:
            cond_ratio, left, right, sup, new_prev_cond_indices = potential_rule
            print('check potential rule',f,cond_ratio,left,right,sup)
            if sup < min_support or cond_ratio <= 1.:
                print('no enough support,skip',sup,cond_ratio)
                
            else:          
                print('add rule',path, f, potential_rule[:-1])  
                valid = True                  
                new_node = RuleNode(f,parent,left,right,cond_ratio,sup)
                parent.add_child(new_node)
                path_copy = path.copy()
                path_copy.append(f)
                if len(path_copy) < max_depth:
                    add_branch_to_rule_tree(new_node,fids_copy,x,target_indices,prev_cond_indices=new_prev_cond_indices,
                                            path=path_copy,grid_num=grid_num,min_support=min_support,max_depth=max_depth,
                                            top_K=top_K,local_x=local_x,search=search,verbose=verbose)
        if not valid:
            print('no valid rule,skip',f)
            fids_copy.remove(f)
            if parent.fid!=-1:
                fids_copy.insert(0,parent.fid)
            add_branch_to_rule_tree(parent,fids_copy,x,target_indices,prev_cond_indices=prev_cond_indices,path=path,grid_num=grid_num,
                                    min_support=min_support,max_depth=max_depth,top_K=top_K,local_x=local_x,search=search,verbose=verbose)

    return


    
def gen_rule_lists_for_one_latent_state(x,z,itemsets_z,zw_pos,thd_h,thd_l,y=None,min_support_pos=500,min_support_neg=2000,
                                    num_grids=20,max_depth=5,local_x=None,top_K=3,feature_types=None,search="ordered",verbose=False):
    comb_z = gen_freq_feature_set(itemsets_z,min_support=min(min_support_pos,min_support_neg),max_len=max_depth)
    comb_z = np.array(comb_z).astype(int)-1
    print('feature set',comb_z)
    min_support_h = min_support_pos if zw_pos else min_support_neg
    min_support_l = min_support_neg if zw_pos else min_support_pos


    rule_dict_higher_z = gen_rule_list_for_one_target(x,comb_z,z>=thd_h,y=y,c=int(zw_pos),min_support=min_support_h,num_grids=num_grids,max_depth=max_depth,
                                                             top_K=top_K,local_x=local_x,feature_types=feature_types,search=search,verbose=verbose)
    rule_dict_lower_z = gen_rule_list_for_one_target(x,comb_z,z<=thd_l,y=y,c=int(1-zw_pos),min_support=min_support_l,num_grids=num_grids,max_depth=max_depth,
                                                             top_K=top_K,local_x=local_x,feature_types=feature_types,search=search,verbose=verbose)
    
    return rule_dict_higher_z,rule_dict_lower_z




def find_pattern_by_latent_state(x,z,itemsets_z,zw_pos,y=None,c=1,num_grids=20,omega=0.1,min_support_pos=500,min_support_neg=2000,
                                 max_depth=4,local_x=None,top_K=3,feature_names=None,feature_types=None,verbose=False):
    thd_h,thd_l = scan_thresholds_for_one_latent_state(z,y==c,zw_pos=zw_pos,omega=omega,min_support_pos=min_support_pos,
                                                       min_support_neg=min_support_neg,num_grids=num_grids)
    print("thd_h",thd_h,"thd_l",thd_l,"pos",zw_pos)
    rule_list_higher_z,rule_list_lower_z = gen_rule_lists_for_one_latent_state(x,z,itemsets_z,zw_pos,thd_h,thd_l,y=y,min_support_pos=min_support_pos,
                                                                               min_support_neg=min_support_neg,num_grids=num_grids,max_depth=max_depth,
                                                                               top_K=top_K,local_x=local_x,feature_names=feature_names,
                                                                               feature_types=feature_types,verbose=verbose) 
    z_rules_dict = {"pos":zw_pos,
                    "thd_h":thd_h,
                    "thd_l":thd_l,
                    "p(z>=thd_h)":np.sum(z>=thd_h)/len(z),
                    "p(z<=thd_l)":np.sum(z<=thd_l)/len(z)}
    z_rules_dict["rule_dict_higher_z"] = rule_list_higher_z
    z_rules_dict["rule_dict_lower_z"] = rule_list_lower_z

    return z_rules_dict


def scan_thresholds_for_one_latent_state(z,Y,zw_pos,omega=0.1,min_support_pos=500,min_support_neg=2000,num_grids=20):
    ## Y is binary
    print("min",z.min(),"max",z.max())
    grids = np.linspace(z.min(),z.max(),num_grids)
    tot_p_true = np.sum(Y==1)/len(Y)
    tot_p_false = np.sum(Y==0)/len(Y)
    splits = []
    igains_h, igains_l= [],[]
    supports_h, supports_l = [],[]
    print('p(y=1)',tot_p_true)
    for g in grids:
        
        if np.sum(Y[z>=g]==1)==0 or np.sum(Y[z>=g]==0)==0:
            continue
        h_pr = (z>=g).sum()/len(Y)  
        if h_pr == 1. or h_pr ==0.:
            continue
            
        # print('split',g) 
        ih,il,sh,sl = calc_split_informgains(g,Y,z,zw_pos,tot_p_true,tot_p_false,h_pr)
        igains_h.append(ih)
        igains_l.append(il)
        supports_h.append(sh)
        supports_l.append(sl)
                
        splits.append(g)
    thd_h = locate_threshold(np.array(igains_h),np.array(supports_h),splits,omega=omega,min_support=min_support_pos)
    thd_l = locate_threshold(np.array(igains_l),np.array(supports_l),splits,omega=omega,min_support=min_support_neg)
        
    
    return thd_h,thd_l


def calc_split_informgains(split,Y,z,zw_pos,tot_p_true,tot_p_false,h_pr):
    
    if zw_pos:
        h_p_true = np.sum(Y[z>=split]==1)/len(Y[z>=split])
        l_p_false = np.sum(Y[z<=split]==0)/len(Y[z<=split])

        igain_h1 = inform_gain(tot_p_true,[h_pr],[np.array([h_p_true])])
        igain_l0 = inform_gain(tot_p_false,[1.-h_pr],[np.array([l_p_false])])  
        support_h1 = np.sum(Y[z>=split]==1)
        support_l0 = np.sum(Y[z<=split]==0)
        return igain_h1,igain_l0,support_h1,support_l0
    else:
        h_p_false = np.sum(Y[z>=split]==0)/len(Y[z>=split])        
        l_p_true = np.sum(Y[z<=split]==1)/len(Y[z<=split])
        
        igain_h0 = (inform_gain(tot_p_false,[h_pr],[np.array([h_p_false])]))
        igain_l1 = (inform_gain(tot_p_true,[1.-h_pr],[np.array([l_p_true])])) 
        support_l1 = np.sum(Y[z<split]==1)
        support_h0 = np.sum(Y[z>=split]==0) 
        return igain_h0,igain_l1,support_h0,support_l1
    
    
def locate_threshold(igains,supports,splits,omega=0.1,min_support=2000):

    if (supports>=min_support).sum()==0:
        print('no enough support')
        return 0.
        
    igains = (igains-igains.min())/(igains.max()-igains.min())
    igains[supports<min_support]=0.
    supports = (supports-supports.min())/(supports.max()-supports.min())
    thd = splits[np.argmax(igains*(1.-omega)+supports*omega)]
    return thd


def entropy(probs):
    return -np.sum(probs*np.log(probs+1e-5))

def inform_gain(p_class_probs,c_probs,c_class_probs):
    H_p = entropy(p_class_probs)
    H_pc = 0.
    for c_pr, c_cl_pr in zip(c_probs, c_class_probs):
        H_c = entropy(c_cl_pr)
        H_pc += c_pr*H_c
        
    return H_p - H_pc


    
def sort_rules(z_rules,input_feature_names,sort_by="cond_prob_y",pos=True,top=3):
    sorted_rules = []
    for zid, zr_dict in z_rules.items():
        new_zr = []
        print(zid,zr_dict['pos'])
        if zr_dict['pos']==pos:
            list_name = 'rule_dict_higher_z'
            p_thd = 'p(z>=thd_h)'
            thd = 'thd_h'
        else:
            list_name = 'rule_dict_lower_z'
            p_thd = 'p(z<=thd_l)'
            thd = 'thd_l'
        ## for dict with only one side    
        if list_name not in zr_dict.keys():
            if 'p(z>=thd_h)' in zr_dict.keys():
                list_name = 'rule_dict_higher_z'  
                p_thd = 'p(z>=thd_h)'
                thd = 'thd_h'
            else:
                list_name = 'rule_dict_lower_z'
                p_thd = 'p(z<=thd_l)'
                thd = 'thd_l'
        
        for rdict in zr_dict[list_name]:
            new_zrd = {}
            new_r = []
            
            for r in rdict["rules"]:
                new_r.append((r[0],input_feature_names[r[0]],r[1],r[2]))

            if len(new_r) == 0:
                print("no rules",zr_dict)
                continue
            new_zrd["rules"] = new_r
            new_zrd['zid'] = zid
            new_zrd[p_thd]=zr_dict[p_thd]
            new_zrd[thd]=zr_dict[thd] 
            new_zrd['pos'] = zr_dict['pos']
            for f in ['cond_prob_y','cond_prob_target','support','ratio_y']:
                new_zrd[f] = rdict[f]

            sorted_rules.append(new_zrd)

    sorted_rules.sort(key=lambda x: x[sort_by], reverse=True)
    return sorted_rules


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


def match_sample_rules(xi,rules_list,time_dim=False,num_latent_per_time=1):
    ret = []
    for rules in rules_list:
        match = []
        if time_dim:
            time_step= int(rules["zid"]/num_latent_per_time)
            #latent_id = int(rules["zid"]%num_latent_per_time)
            xi_t = xi[time_step,:]
        else:
            xi_t = xi
        for r in rules["rules"]:
            fval = xi_t[r[0]]
            if not op_map[r[-2]](fval,r[-1]):
                # match = False
                break
            else:
                match.append(r)
        if len(match) > 0:            
            ret.append(match)
    return ret


def get_sample_latent_state_thresholds(zi,y,z,zw_pos,c=1,num_grids=20,omega=0.1,min_support_pos=500,min_support_neg=2000):
    thd_h,thd_l = scan_thresholds_for_one_latent_state(z,y==c,zw_pos=zw_pos,omega=omega,min_support_pos=min_support_pos,
                                                       min_support_neg=min_support_neg,num_grids=num_grids)
    print("raw thd",thd_h,thd_l)
    if zi >= thd_h:
        return thd_h,True
    elif zi <= thd_l:
        return thd_l,False
    else:
        tot_p_true = np.sum(y==1)/len(y)
        tot_p_false = np.sum(y==0)/len(y)
        h_pr = (z>=zi).sum()/len(y)
        ih,il,sh,sl = calc_split_informgains(zi,y,z,zw_pos,tot_p_true,tot_p_false,h_pr)
        sign = ih > il
        return zi, sign


