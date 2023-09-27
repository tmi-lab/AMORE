
import numpy as np
import operator


op_map = {'>=':operator.ge,'>':operator.gt,'<=':operator.le,'<':operator.lt,'==':operator.eq}


from .itemsets_miner import *
from .RuleGrowth_tree import RuleTree,RuleNode



def merge_rule(rule_conj):
    r_dict = {}
    del_list = []
    #print('start merge',rule_conj)
    for i,r in enumerate(rule_conj):
        #print(i,r)
        rid = r[0]
        r_dict[rid] = r_dict.get(rid,None)
        #print(rid,r_dict[rid])
        if r_dict[rid] is None:
            r_dict[rid] = [i]
        else:
            #print('found same rid',r_dict[rid],rid)
            new_r = None
            for jj in r_dict[rid]:
                #print(jj,rule_conj[jj],r)
                new_r = merge_subspace(r,rule_conj[jj])
                if new_r is not None:
                    ## merged to a previous subspace
                    rule_conj[jj] = new_r
                    #print('new',new_r)
                    #del rule_conj[i] ## need to fix
                    del_list.append(i)
                    break
                    
            if new_r is None:
                r_dict[rid].append(i)
    if len(del_list)>0:
         rule_conj = [r for i, r in enumerate(rule_conj) if i not in del_list]
    #print('after merge',rule_conj)                
    return rule_conj


def merge_subspace(r1,r2):
    assert(r1[0]==r2[0])
    if '>' in r1[1] and '>' in r2[1]:
        a = np.argmax([r1[2],r2[2]])
        
    elif '<' in r1[1] and '<' in r2[1]:
        a = np.argmin([r1[2],r2[2]]) 
    elif r1[1] == r2[1] and r1[2] == r2[2]:
        return r2
    else:
        return None
    
    return r1 if a==0 else r2
    
    
def find_suport(X,rule_conj):
    indices = np.ones(X.shape[0]).astype(bool)
    for r in rule_conj:
        indices = indices & find_support_one_rule(X,r)
    return indices


def find_support_one_rule(X,rule):
    fid = rule[0]
    return op_map[rule[1]](X[...,fid],rule[2])


def search_feature_rule_one_side(f_val,z_indices,grids,prev_cond_indices=None,min_support=2000,local_x=None):
    ratios,supports = scan_feature_cond_prob_ratio(f_val,z_indices,grids,prev_cond_indices=prev_cond_indices) 
    cond_ratio,left_id,right_id,sup = raise_feature_range(f_val,grids,ratios,supports,z_indices,
                                                          prev_cond_indices=prev_cond_indices,
                                                          min_support=min_support,local_x=local_x)
    if cond_ratio < 1.:
        return 
    return (cond_ratio,grids[left_id],grids[right_id+1],sup)



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


def raise_feature_range(f_val,grids,ratios,supports,z_indices,prev_cond_indices=None,min_support=2000,local_x=None):
    if local_x is None:
        sid = np.argmax(ratios)
    else:
        sid = np.arange(len(grids)-1)[(grids[:-1] -local_x.numpy())<=0][-1]
        #print("check local",sid,local_x,grids[sid],grids[sid+1])
    if prev_cond_indices is None:
        sup = supports[sid]
    else:
        sup = np.sum((f_val[prev_cond_indices]>=grids[sid])&(f_val[prev_cond_indices]<=grids[sid+1]))
    explore = False if local_x is None else True
    ratio,left_sid,right_sid,sup = merge_feature_range(sid,sup,f_val,grids,ratios,z_indices,
                                                       prev_cond_indices=prev_cond_indices,
                                                       min_support=min_support,explore=explore)
         
    ### check the support of the new range, if local_x is not None, check if local_x is in the new range
    if sup < min_support:
        ratios[left_sid:right_sid+1]=0.
        if np.max(ratios) <= 1. or (local_x is not None and (local_x>=grids[left_sid] and local_x<=grids[right_sid+1])):
            ## no range can be raised if all ratios are smaller than 1 
            ## or local_x is in the range without enough support
            return -1.,-1,-1,-1
        else:
            ratio,left_sid,right_sid,sup = raise_feature_range(f_val,grids,ratios,supports,z_indices,prev_cond_indices=prev_cond_indices,min_support=min_support)
    if local_x is not None:
        print("check local",local_x,grids[left_sid],grids[right_sid+1])
    return ratio,left_sid,right_sid,sup


def merge_feature_range(sid,sup,f_val,grids,ratios,z_indices,prev_cond_indices=None,min_support=2000,explore = True):

    left_sid = sid
    right_sid = sid
    old_r = ratios[sid]
    while sup < min_support or explore:
        left = left_sid - 1
        right = right_sid + 1
        left_r = ratios[left] if left >= 0 else 0.
        right_r = ratios[right] if right < len(ratios) else 0.
        
        ## merge neighbor grid with 0 support
        if left_r == -1.:
            merge = left
        elif right_r == -1.:
            merge = right            
        ## merge neighbor grid with ratio larger than 1
        elif left_r >= 1. and left_r >= right_r:
            merge = left 
        elif right_r >= 1. and right_r > left_r:
            merge = right
        else:
            explore = False
            break

        new_r,new_sup = calc_cond_ratio(f_val,grids[min(merge,left_sid)],grids[max(merge,right_sid)+1],
                                        z_indices,prev_cond_indices)

        old_r = new_r
        left_sid = min(merge,left_sid)
        right_sid = max(merge,right_sid)
        sup = new_sup
    return old_r,left_sid,right_sid,sup
    


def calc_cond_ratio(f_val,left_val,right_val,z_indices,prev_cond_indices=None):
    if prev_cond_indices is not None:
        cond_indices = z_indices & prev_cond_indices
    else:
        cond_indices = z_indices
        prev_cond_indices = np.ones_like(f_val).astype(bool)
    #print('prev cond indices',np.sum(prev_cond_indices))
    support = np.sum((f_val[prev_cond_indices]>=left_val)&(f_val[prev_cond_indices]<=right_val))
    if support == 0:
        ratio = -1.
        return ratio,support
    
    # dom = support/len(f_val[prev_cond_indices])
    # ratio = (np.sum((f_val[cond_indices]>=left_val)&(f_val[cond_indices]<=right_val))/np.sum(cond_indices))/dom
    dom = np.sum(cond_indices)/np.sum(prev_cond_indices)
    ratio = (np.sum((f_val[cond_indices]>=left_val)&(f_val[cond_indices]<=right_val))/support)/dom

    return ratio,support


def entropy(probs):
    return -np.sum(probs*np.log(probs+1e-5))

def inform_gain(p_class_probs,c_probs,c_class_probs):
    H_p = entropy(p_class_probs)
    H_pc = 0.
    for c_pr, c_cl_pr in zip(c_probs, c_class_probs):
        H_c = entropy(c_cl_pr)
        H_pc += c_pr*H_c
        
    return H_p - H_pc

   

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



def search_potential_rules_one_side(fids,x,z_indices,min_support=2000,grid_num=100,prev_cond_indices=None,sort=True,local_x=None):
    potential_rules = []
    for f in fids:
        #print('feature',f)
        f_val = x[:,int(f)]
        grids = np.linspace(f_val.min(),f_val.max(),grid_num) 
        lx = None if local_x is None else local_x[int(f)]       
        potential_rule = search_feature_rule_one_side(f_val,z_indices,grids,prev_cond_indices,min_support,local_x=lx)
        if potential_rule is None:
            continue
        potential_rules.append({"fid":f,"rule":potential_rule})
        #print("add potential rule",potential_rule)
    if sort:
        potential_rules.sort(key=lambda x: x["rule"][0],reverse=True)
    print("num potential rules",len(potential_rules))
    return potential_rules



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

    
def gen_rule_lists_for_one_latent_state(x,z,itemsets_z,zw_pos,thd_h,thd_l,min_support_pos=500,min_support_neg=2000,
                                        num_grids=20,max_depth=5,local_x=None):
    comb_z = gen_freq_feature_set(itemsets_z,min_support=min(min_support_pos,min_support_neg),max_len=max_depth)
    comb_z = np.array(comb_z).astype(int)-1
    print('feature comb',comb_z)
    min_support_h = min_support_pos if zw_pos else min_support_neg
    min_support_l = min_support_neg if zw_pos else min_support_pos

    # rule_dict_higher_z = gen_rule_list_for_one_target(x,comb_z,z>=thd_h,min_support=min_support_h,num_grids=num_grids,max_depth=max_depth,local_x=local_x)
    # rule_dict_lower_z = gen_rule_list_for_one_target(x,comb_z,z<=thd_l,min_support=min_support_l,num_grids=num_grids,max_depth=max_depth,local_x=local_x)

    rule_tree_lower_z = build_rule_tree_one_side(list(comb_z),x,z<=thd_l,grid_num=num_grids,min_support=min_support_l,
                                                 max_depth=max_depth,local_x=local_x)
    rule_tree_higher_z = build_rule_tree_one_side(list(comb_z),x,z>=thd_h,grid_num=num_grids,min_support=min_support_h,
                                                  max_depth=max_depth,local_x=local_x)

    if rule_tree_higher_z is None:
        rule_dict_higher_z = {}
    else:
        _,rule_dict_higher_z = rule_tree_higher_z.get_rule_dict()
    if rule_tree_lower_z is None:
        rule_dict_lower_z = {}
    else:
        _,rule_dict_lower_z = rule_tree_lower_z.get_rule_dict()
        
    rule_dict_higher_z = remove_duplicate_rules(rule_dict_higher_z)
    rule_dict_lower_z = remove_duplicate_rules(rule_dict_lower_z)
    
    return rule_dict_higher_z,rule_dict_lower_z


def gen_rule_list_for_one_target_greedy(x,comb_z,zids,min_support=500,num_grids=20,max_depth=5,local_x=None):
    
    rule_tree_z = build_rule_tree_one_side(list(comb_z),x,zids,grid_num=num_grids,min_support=min_support,
                                                 max_depth=max_depth,local_x=local_x)
    if rule_tree_z is None:
        rule_dict_z = {}
    else:
        _,rule_dict_z = rule_tree_z.get_rule_dict()
    
    rule_dict_z = remove_duplicate_rules(rule_dict_z)
    
    return rule_dict_z


def gen_rule_list_for_one_target(x,fids,target_indices,min_support=500,num_grids=20,max_depth=5,local_x=None):
    
    fids_copy = list(fids).copy()
    prev_cond_indices = None
    rule_dict = {}
    while len(fids_copy)>0:
        for f in fids_copy:
            ratio,left,right,sup,prev_cond_indices = add_one_rule(x,f,target_indices,prev_cond_indices=prev_cond_indices,num_grids=num_grids,
                            min_support=min_support,max_depth=max_depth,local_x=local_x)
            # potential_rule = search_feature_rule_interval(f_val,cond_indices,grids,prev_cond_indices,min_support,local_x=lx)
            # if potential_rule is None:
            #     continue
            # potential_rules.append({"fid":f,"rule":potential_rule})
            rule_dict[f] = {"fid":f,"rule":(ratio,left,right,sup)}
    rule_dict = remove_duplicate_rules(rule_dict)
    
    return rule_dict


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
        grids[s] = merge
        ratios[s] = ratios[merge]
    new_grids = np.unique(grids)
    return new_grids
                    

def find_peaks(ratios):
    peaks = []
    for i in range(1,len(ratios)-1):
        if ratios[i]>ratios[i-1] and ratios[i]>ratios[i+1] and ratios[i]>1.:
            peaks.append(i)
    return peaks


def search_feature_rule_interval(f_val,target_indices,grids,prev_cond_indices=None,min_support=2000,local_x=None):
    ratios,supports = scan_feature_cond_prob_ratio(f_val,target_indices,grids,prev_cond_indices=prev_cond_indices) 
    cond_ratio,left_id,right_id,sup = raise_feature_range(f_val,grids,ratios,supports,target_indices,
                                                          prev_cond_indices=prev_cond_indices,
                                                          min_support=min_support,local_x=local_x)
    if cond_ratio < 1.:
        return 
    return (cond_ratio,grids[left_id],grids[right_id+1],sup)

# def search_potential_rules_one_side(fids,x,z_indices,min_support=2000,grid_num=100,prev_cond_indices=None,sort=True,local_x=None):
#     potential_rules = []
#     for f in fids:
#         #print('feature',f)
#         f_val = x[:,int(f)]
#         grids = np.linspace(f_val.min(),f_val.max(),grid_num) 
#         lx = None if local_x is None else local_x[int(f)]       
#         potential_rule = search_feature_rule_one_side(f_val,z_indices,grids,prev_cond_indices,min_support,local_x=lx)
#         if potential_rule is None:
#             continue
#         potential_rules.append({"fid":f,"rule":potential_rule})
#         #print("add potential rule",potential_rule)
#     if sort:
#         potential_rules.sort(key=lambda x: x["rule"][0],reverse=True)
#     print("num potential rules",len(potential_rules))
#     return potential_rules


def add_one_rule(x,f,target_indices,prev_cond_indices=None,num_grids=20,
                            min_support=2000,max_depth=4,local_x=None):
    
    f_val = x[:,int(f)]
    grids = np.linspace(f_val.min(),f_val.max(),num_grids) 
    ratios,supports = scan_feature_cond_prob_ratio(f_val,target_indices,grids,prev_cond_indices=prev_cond_indices)
    grids = merge_empty_grids(grids,ratios,supports)
    ratios,supports = scan_feature_cond_prob_ratio(f_val,target_indices,grids,prev_cond_indices=prev_cond_indices)
    peaks = find_peaks(grids,ratios)
    lx = None if local_x is None else local_x[int(f)]  
    
    op_ratio = 0.
    op_left = 0.
    op_right = 0.
    op_sup = 0.
    for gid in peaks:
        cond_ratio,left,right,sup = raise_feature_range(f_val,grids,gid,target_indices,prev_cond_indices=prev_cond_indices,min_support=min_support,local_x=lx)     


        if sup < min_support or cond_ratio < 1. or cond_ratio < op_ratio:
            print('not qualified',sup,cond_ratio)
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
                op_ratio = cond_ratio
                op_left = left
                op_right = right
                op_sup = sup
                
    return op_ratio,op_left,op_right,op_sup, new_prev_cond_indices

# def obtain_rule_lists_for_one_latent_state(x,z,itemsets_z,zw_pos,z_left,z_right,min_support_pos=500,min_support_neg=2000,num_grids=20,max_depth=5):
#     comb_z = gen_freq_feature_set(itemsets_z,min_support=min(min_support_pos,min_support_neg),max_len=max_depth)
#     comb_z = np.array(comb_z).astype(int)-1
#     print('feature comb',comb_z)
#     min_support = min_support_pos if zw_pos else min_support_neg
#     #min_support_l = min_support_neg if zw_pos else min_support_pos

#     rule_tree_z = build_rule_tree_one_range(list(comb_z),x,(z<=z_right)&(z>=z_left),
#                                             grid_num=num_grids,min_support=min_support,max_depth=max_depth)
#     #rule_tree_higher_z = build_rule_tree_one_side(list(comb_z),x,z>=thd_h,grid_num=num_grids,min_support=min_support_h,max_depth=max_depth)

#     if rule_tree_z is None:
#         rule_dict_z = {}
#     else:
#         _,rule_dict_z = rule_tree_z.get_rule_dict()
        
#     rule_dict_z = remove_duplicate_rules(rule_dict_z)
    
#     return rule_dict_z



# def gen_z_rules_for_class_output(z,y_prob,itemsets_y,thd_h,thd_l,min_support_pos=500,min_support_neg=2000,num_grids=20,max_depth=5):

#     comb_z = gen_freq_feature_set(itemsets_y,min_support=min(min_support_pos,min_support_neg),max_len=max_depth)
#     comb_z = np.array(comb_z).astype(int)-1
#     print('latent comb',comb_z)
#     min_support_h = min_support_pos 
#     min_support_l = min_support_neg 

#     rule_tree_lower_y = build_rule_tree_one_range(list(comb_z),z,y_prob<=thd_l,grid_num=num_grids,min_support=min_support_l,max_depth=max_depth)
#     rule_tree_higher_y = build_rule_tree_one_range(list(comb_z),z,y_prob>=thd_h,grid_num=num_grids,min_support=min_support_h,max_depth=max_depth)

#     if rule_tree_higher_y is None:
#         rule_dict_higher_y = {}
#     else:
#         _,rule_dict_higher_y = rule_tree_higher_y.get_rule_dict()
#     if rule_tree_lower_y is None:
#         rule_dict_lower_y = {}
#     else:
#         _,rule_dict_lower_y = rule_tree_lower_y.get_rule_dict()
        
#     rule_dict_higher_y = remove_duplicate_rules(rule_dict_higher_y)
#     rule_dict_lower_y = remove_duplicate_rules(rule_dict_lower_y)
    
#     return rule_dict_higher_y,rule_dict_lower_y



# def obtain_z_intervals(rule_dict_y,z,y,c,K=1):
#     d_rules = []    
#     for p,rules in rule_dict_y.items():
#         d_rules.append(display_rules(rules["rules"],z,y,None,c=c))        
#     d_rules.sort(key=lambda x: x["cond_prob_y"], reverse=True)
#     z_intervals = []
#     for k in range(K):
#         if k >= len(d_rules):
#             break
#         for r in d_rules[k]["rules"]:
#             z_intervals.append(r)
#     return z_intervals


# def find_class_pattern_by_latent_state(x,y,y_prob,itemsets_y,Z,itemsets_Z,ZW_pos,num_grids=20,omega=0.1,min_support_pos=500,min_support_neg=2000,max_depth=4):
#     ## find threshold for y_prob
#     thd_h,thd_l = scan_thresholds_for_one_latent_state(y_prob,y==1,zw_pos=True,omega=omega,min_support_pos=min_support_pos,
#                                                        min_support_neg=min_support_neg,num_grids=num_grids)
#     print("thd_h",thd_h,"thd_l",thd_l)
#     rule_list_higher_y,rule_list_lower_y = gen_z_rules_for_class_output(Z,y_prob,itemsets_y,True,thd_h,thd_l,min_support_pos=min_support_pos,
#                                                                                min_support_neg=min_support_neg,num_grids=num_grids,max_depth=max_depth) 
#     higher_y_z_intervals = obtain_z_intervals(rule_list_higher_y,Z,y,c=1,K=1)
#     lower_y_z_intervals = obtain_z_intervals(rule_list_lower_y,Z,y,c=0,K=1)
    
#     higher_y_z_rules = []
#     for z_interval in higher_y_z_intervals:
#         zid = z_interval[0]
#         z_left = z_interval[1]
#         z_right = z_interval[2]
#         z_rules = obtain_rule_lists_for_one_latent_state(x,Z[:,zid],itemsets_Z[zid],True,z_left,z_right,min_support_pos=min_support_pos,
#                                     min_support_neg=min_support_neg,num_grids=num_grids,max_depth=max_depth)
#         higher_y_z_rules.append(display_rules(z_rules["rules"],x,y,zids=(Z[zid]>=z_left)&(Z[zid]<=z_right),c=1))
#     higher_y_z_rules.sort(key=lambda x: x["cond_prob_y"], reverse=True)
    
#     lower_y_z_rules = []
#     for z_interval in lower_y_z_intervals:
#         zid = z_interval[0]
#         z_left = z_interval[1]
#         z_right = z_interval[2]
#         z_rules = obtain_rule_lists_for_one_latent_state(x,Z[:,zid],itemsets_Z[zid],False,z_left,z_right,min_support_pos=min_support_pos,
#                                     min_support_neg=min_support_neg,num_grids=num_grids,max_depth=max_depth)
#         lower_y_z_rules.append(display_rules(z_rules["rules"],x,y,zids=(Z[zid]>=z_left)&(Z[zid]<=z_right),c=0))
#     lower_y_z_rules.sort(key=lambda x: x["cond_prob_y"], reverse=True)
    
#     z_rules_dict = {
#                     "thd_h":thd_h,
#                     "thd_l":thd_l,
#                     "p(y_prob>=thd_h)":np.sum(y_prob>=thd_h)/len(y),
#                     "p(y_prob<=thd_l)":np.sum(y_prob<=thd_l)/len(y)}
    
#     z_rules_dict["rule_dict_higher_y"] = higher_y_z_rules
#     z_rules_dict["rule_dict_lower_y"] = lower_y_z_rules
        

#     return z_rules_dict



# def build_rule_tree_one_range(fids,x,z_indices,grid_num=20,min_support=2000,max_depth=4):
#     print("build_rule_tree_one_side")
#     rule_tree = RuleTree(min_support=min_support)
#     add_branch_to_rule_tree(rule_tree.root,fids,x,z_indices,prev_cond_indices=None,grid_num=grid_num,
#                                 min_support=min_support,max_depth=max_depth)
#     return rule_tree




def build_rule_tree_one_side(fids,x,z_indices,grid_num=20,min_support=2000,max_depth=4,local_x=None):
    print("build_rule_tree_one_side")
    rule_tree = RuleTree(min_support=min_support)
    add_branch_to_rule_tree(rule_tree.root,fids,x,z_indices,prev_cond_indices=None,grid_num=grid_num,
                                min_support=min_support,max_depth=max_depth,local_x=local_x)
    return rule_tree

        
def transform_rules(rule):    
    new_r = {}
    new_r["fid"] = rule["fid"]
    new_r["left"] = rule["rule"][1]
    new_r["right"] = rule["rule"][2]
    new_r["score"] = rule["rule"][0]
    new_r["support"] = rule["rule"][3]
    return new_r    
    

def add_branch_to_rule_tree(parent,fids,x,z_indices,prev_cond_indices=None,grid_num=20,
                            min_support=2000,max_depth=4,local_x=None):
    fids_copy = fids.copy()
    if parent.fid != -1:
        fids_copy.remove(parent.fid)
    if len(fids_copy)>0:
        potential_rules = search_potential_rules_one_side(fids_copy,x,z_indices,min_support=min_support,grid_num=grid_num,
                                                          prev_cond_indices=prev_cond_indices,local_x=local_x)        
        if len(potential_rules)==0:
            return        
        for potential_rule in potential_rules:
            f = int(potential_rule["fid"])
            if len(potential_rule["rule"])==0:
                continue

            cond_ratio, left, right, sup = potential_rule["rule"]
            if sup < min_support or cond_ratio < 1.:
                print('no enough support,exit',sup)
                continue
            else:          
                f_val = x[:,f]
                valid = True
                if left == f_val.min() and right == f_val.max():
                    valid = False
                if valid:
                    if prev_cond_indices is None:
                        new_prev_cond_indices = (f_val>=left)&(f_val<=right)
                    else:
                        new_prev_cond_indices = prev_cond_indices & ((f_val>=left)&(f_val<=right))
                    #print('cond indices',np.sum(prev_cond_indices))
                    print('add rule',potential_rule)                    
                    new_node = RuleNode(f,parent,left,right,cond_ratio,sup)
                    parent.add_child(new_node)
                    add_branch_to_rule_tree(new_node,fids_copy,x,z_indices,prev_cond_indices=new_prev_cond_indices,
                                            grid_num=grid_num,min_support=min_support,max_depth=max_depth,local_x=local_x)
    else:
        print('reach max depth')
    return


def target_prob_with_rules(rule_list,x,zids=None,y=None,c=1,verbose=False):
    mask = np.ones_like(y).astype(bool)
    for r in rule_list:
        mask = mask & op_map[r[1]](x[...,int(r[0])],r[2])
        if verbose:
            print(r,np.sum(mask))
    if y is None:
        return np.sum(mask & zids)/np.sum(mask),np.nan,np.nan,np.sum(mask)
    if zids is None:
        return np.nan,np.sum(y[mask]==c)/np.sum(mask),np.sum(y[mask]==c)/np.sum(y==c),np.sum(mask)
    
    return np.sum(mask & zids)/np.sum(mask), np.sum(y[mask]==c)/np.sum(mask),np.sum(y[mask]==c)/np.sum(y==c),np.sum(mask)


def display_rules(rules,x,y,zids,c,verbose=False):
    print("display",c)
    for r in rules:
        ## remove useless rules
        if op_map[r[1]](x[...,int(r[0])],r[2]).sum() == len(x):
            rules.remove(r)
    h_cond_prob_z,h_cond_prob_y, h_ratio_y,h_sup = target_prob_with_rules(rules,x,zids=zids,y=y,c=c,verbose=verbose) 
    return {"rules":rules,
            "cond_prob_z":h_cond_prob_z,
            "cond_prob_y":h_cond_prob_y,
            "ratio_y":h_ratio_y,
            "support":h_sup}



           
def find_pattern_by_latent_state(x,y,z,itemsets_z,zw_pos,c=1,num_grids=20,omega=0.1,min_support_pos=500,min_support_neg=2000,max_depth=4):
    thd_h,thd_l = scan_thresholds_for_one_latent_state(z,y==c,zw_pos=zw_pos,omega=omega,min_support_pos=min_support_pos,
                                                       min_support_neg=min_support_neg,num_grids=num_grids)
    print("thd_h",thd_h,"thd_l",thd_l,"pos",zw_pos)
    rule_list_higher_z,rule_list_lower_z = gen_rule_lists_for_one_latent_state(x,z,itemsets_z,zw_pos,thd_h,thd_l,min_support_pos=min_support_pos,
                                                                               min_support_neg=min_support_neg,num_grids=num_grids,max_depth=max_depth) 
    z_rules_dict = {"pos":zw_pos,
                    "thd_h":thd_h,
                    "thd_l":thd_l,
                    "p(z>=thd_h)":np.sum(z>=thd_h)/len(z),
                    "p(z<=thd_l)":np.sum(z<=thd_l)/len(z)}
    z_rules_dict["rule_dict_higher_z"] = {}
    z_rules_dict["rule_dict_lower_z"] = {}
    for p,rules in rule_list_higher_z.items():
        z_rules_dict["rule_dict_higher_z"][p] = display_rules(rules["rules"],x,y,zids=z>=thd_h,c=int(zw_pos))
        if z_rules_dict["rule_dict_higher_z"][p]['support'] < min_support_pos:
            
            print("#######wrong support#########",z_rules_dict["rule_dict_higher_z"][p]['support'])
            print("check x",x[:3,:4])
            print("check y",y.sum(),"check z",(z>=thd_h).sum())
            print(p,rules)
            display_rules(rules["rules"],x,y,zids=z>=thd_h,c=int(zw_pos),verbose=True)
    
    for p,rules in rule_list_lower_z.items():
        z_rules_dict["rule_dict_lower_z"][p] = display_rules(rules["rules"],x,y,zids=z<=thd_l,c=int(1-zw_pos))
        if z_rules_dict["rule_dict_lower_z"][p]['support'] < min_support_pos:
            print("check x",x[:3,:4])
            print("#######wrong support#########",z_rules_dict["rule_dict_lower_z"][p]['support'])
            print("check x",x[:3,:4])
            print("check y",y.sum(),"check z",(z<=thd_l).sum())
            print(p,rules)
            display_rules(rules["rules"],x,y,zids=z<=thd_l,c=int(1-zw_pos),verbose=True)
    return z_rules_dict


def find_pattern_by_sample_latent_state(xi,zi,x,y,z,itemsets_z,zw_pos,c=1,num_grids=20,omega=0.1,min_support_pos=500,min_support_neg=2000,max_depth=4):

    thd,sign = get_sample_latent_state_thresholds(zi,y,z,zw_pos,c=c,num_grids=num_grids,omega=omega,min_support_pos=min_support_pos,min_support_neg=min_support_neg)
    print("thd",thd,"pos",zw_pos)
    zids = z>=thd if sign else z<=thd
    min_support = min_support_pos if (zw_pos*zi>0) else min_support_neg
    
    comb_z = gen_freq_feature_set(itemsets_z,min_support=min(min_support_pos,min_support_neg),max_len=max_depth)
    comb_z = np.array(comb_z).astype(int)-1
    print('feature comb',comb_z)
    
    raw_rules_dict_z = gen_rule_list_for_one_target(x,comb_z,zids,min_support=min_support,num_grids=num_grids,max_depth=max_depth,local_x=xi)
    cond_z = "p(z>=thd_h)" if sign else "p(z<=thd_l)"
    thd_name = "thd_h" if sign else "thd_l"
    z_rules_dict = {"pos":zw_pos,
                    thd_name:thd,
                    cond_z:np.sum(zids)/len(z),}
    dname = "rule_dict_higher_z" if sign else "rule_dict_lower_z"
    z_rules_dict[dname] = {}
    for p,rules in raw_rules_dict_z.items():
        z_rules_dict[dname][p] = display_rules(rules["rules"],x,y,zids=zids,c=int(zw_pos*zi>0))
        if z_rules_dict[dname][p]['support'] < min_support:            
            print("#######wrong support#########",z_rules_dict["rule_dict_higher_z"][p]['support'])
            print("check x",x[:3,:4])
            print("check y",y.sum(),"check z",(zids).sum())
            print(p,rules)
            display_rules(rules["rules"],x,y,zids=zids,c=int(zw_pos*zi>0),verbose=True)    
    
    return z_rules_dict


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
        # c = int(zw_pos*zi>0)
        # sign = (y[z>=zi]==c).sum() > (y[z<=zi]==c).sum()
        return zi, sign
        
def match_sample_rules(xi,rules_list,time_dim=False,num_latent_per_time=1):

    for rules in rules_list:
        match = True
        if time_dim:
            time_step= int(rules["zid"]/num_latent_per_time)
            #latent_id = int(rules["zid"]%num_latent_per_time)
            xi_t = xi[time_step,:]
        else:
            xi_t = xi
        for r in rules["rules"]:
            fval = xi_t[r[0]]
            if not op_map[r[-2]](fval,r[-1]):
                match = False
                break
        if match:
            return rules
        else:
            return None
    
    
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

        for path,rules_dict in zr_dict[list_name].items():
            #print(path,rules_dict)
            new_zrd = {}
            new_r = []
            for r in rules_dict["rules"]:
                new_r.append((r[0],input_feature_names[r[0]],r[1],r[2]))
            new_zrd["rules"] = new_r
            new_zrd['zid'] = zid
            new_zrd[p_thd]=zr_dict[p_thd]
            new_zrd[thd]=zr_dict[thd] 
            for f in ['cond_prob_y','cond_prob_z','support','ratio_y']:
                new_zrd[f] = rules_dict[f]

            new_zr.append(new_zrd)

            
        new_zr.sort(key=lambda x: x[sort_by], reverse=True)
        sorted_rules = sorted_rules + new_zr[:top]

    sorted_rules.sort(key=lambda x: x[sort_by], reverse=True)
    return sorted_rules