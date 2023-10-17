
import ast
import numpy as np
from collections.abc import Iterator
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeRegressor
from .rule_pattern_miner import op_map


def obtain_rule_lists_from_DT(dtree,max_depth,x,y,z,feature_ids,input_feature_names,c=1):
    tree_dict = {}
    tree_text = export_text(dtree,decimals=3)
    lines = tree_text.split('\n')
    path = np.zeros(max_depth)-1
    rule_list, rule_value_list, rule_support_list = [],[],[]
    new_lines = []
    feature_ids = np.array(feature_ids).astype(int)
    for l, line in enumerate(lines):
        # Split the line into node depth and node description
        description = line.strip().split('|')
        depth = len(description)-1
        description = description[-1].split(' ')[1:]
    
        for i,ds in enumerate(description):

            if 'feature' in ds:
                xid = int(ds.split('_')[-1])
                path[depth-1] = xid
                path[depth:] = -1

                description[i] = input_feature_names[feature_ids[xid]]
                new_line = line.replace(ds,input_feature_names[feature_ids[xid]])
                new_lines.append(new_line)
                tree_dict[tuple(path[:depth])] = (feature_ids[xid],description[1],float(description[-1]))

                if (path==-1).sum()==0 or ((l+1<len(lines)) and (('value' in lines[l+1]) or ('class' in lines[l+1]))):
                    #print('complete one path')
                    rule_conj = []
                    for i in range(sum(path>-1)):
                        rule_conj.append(tree_dict[tuple(path[:i+1])])
                    rule_conj = merge_rule(rule_conj) 
                    support = find_suport(x,rule_conj)
                    # rule_support_list.append((support.sum(),np.abs(z[support].mean())/z[support].std(),(y[support]==1).sum()/support.sum()))
                    rule_support_list.append((support.sum(),z[support].sum()/support.sum(),(y[support]==c).sum()/support.sum()))

                    rule_list.append(rule_conj)


            if 'value' in ds or 'class' in ds:
                
                val = ast.literal_eval(description[-1])
                if isinstance(val, Iterator):
                    val = val[0]
                rule_value_list.append(val)
                line = line+(' cond_prob_target:'+str(rule_support_list[-1][1].round(3))+' cond_prob_y:'+str(rule_support_list[-1][2].round(3)))
                #print(line)
                new_lines.append(line)
                
    return rule_list, rule_value_list, rule_support_list, new_lines


def select_rule_list(rule_support_list,prob_low_th=0.01,prob_high_th=0.15):
    rule_support1 = [r[1] for r in rule_support_list]
    rule_support2 = [r[2] for r in rule_support_list]

    select=[]
    for i in range(len(rule_support_list)):
        if (rule_support1[i] > prob_high_th or rule_support1[i] < prob_low_th) and (rule_support2[i] > prob_high_th or rule_support2[i] < prob_low_th):
            select.append(i)
    select = np.array(select)
            
    return select


def gen_rules_by_DTree(x,y,z,fids,input_feature_names,snr_th=0.5,prob_low_th=0.02,prob_high_th=0.15,max_depth=5,min_samples_leaf=500):
    dtr = DecisionTreeRegressor(criterion='absolute_error',min_samples_leaf=min_samples_leaf,max_depth=max_depth)
    dtr.fit(x[:,fids],y=z)
    
    rule_list, rule_value_list, rule_support_list, new_lines = obtain_rule_lists_from_DT(dtr,max_depth,x,y,z,fids,input_feature_names)
    selected_ids = select_rule_list(rule_support_list,snr_th=snr_th,prob_low_th=prob_low_th,prob_high_th=prob_high_th)
    select = [[],[],[]]
    for s in selected_ids:
        select[0].append(rule_list[s])
        select[1].append(rule_value_list[s])
        select[2].append(rule_support_list[s])
        print('#################')
        print(rule_list[s])
        print('value',rule_value_list[s],'SNR',rule_support_list[s][1].round(3),'prob',rule_support_list[s][2].round(3),'size',rule_support_list[s][0])
        for r in rule_list[s]:
            print(input_feature_names[r[0]],r[1],r[2])
    return select, dtr, new_lines

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