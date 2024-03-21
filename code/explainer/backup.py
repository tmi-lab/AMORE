def add_branch_to_rule_tree_bk(parent,fids,x,target_indices,prev_cond_indices=None,path=[],grid_num=20,
                            min_support=2000,max_depth=4,top_K=3,local_x=None,search="greedy",verbose=False):
    fids_copy = fids.copy()
    N = len(x)
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
                potential_rules = add_potential_rules(x,f,target_indices,prev_cond_indices=prev_cond_indices,
                                                      num_grids=grid_num,min_support=min_support,top_K=top_K,
                                                      local_x=local_x,verbose=verbose) 
                if len(potential_rules) == 0:
                    continue
                # tot_r = sum([r[0] for r in potential_rules])
                # tot_s = np.bitwise_or.reduce([r[-1] for r in potential_rules])
                # tot_s = np.sum(tot_s)/N
                if best_r < potential_rules[0][0]:
                    best_r = potential_rules[0][0]
                    best_f = f
                    best_potential_rules = potential_rules   
                        
            print('best rule',best_f,best_r,best_potential_rules)
            f = best_f
            potential_rules = best_potential_rules
            
            if best_r<=1.00001:
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
        if not valid and search == "ordered":
            print('no valid rule,skip',f)
            fids_copy.remove(f)
            if parent.fid!=-1:
                fids_copy.insert(0,parent.fid)
            add_branch_to_rule_tree(parent,fids_copy,x,target_indices,prev_cond_indices=prev_cond_indices,path=path,grid_num=grid_num,
                                    min_support=min_support,max_depth=max_depth,top_K=top_K,local_x=local_x,search=search,verbose=verbose)

    return
