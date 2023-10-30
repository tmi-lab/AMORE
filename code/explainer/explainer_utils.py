import pandas as pd
import numpy as np
import torch
from .integrad import integrad

def cosine_sim(a, b,dims):
    sim = (a*b).sum(dim=dims)/(torch.sqrt(torch.sum(a**2,dim=dims))*torch.sqrt(torch.sum(b**2,dim=dims)))
    return sim


def data_to_hidden_states(data,model,times,**kwargs):
    coeffs, _, lengths, __ = data
    zt = model.hidden_state(times, coeffs, lengths,**kwargs)
    return zt.detach()

def data_to_prediction(data,model,times,logits=True,**kwargs):
    coeffs, _, lengths, sinput = data
    pred_y = model(times, coeffs, lengths,side_input=sinput,**kwargs)
    if not logits:
        ## return labels
        if len(pred_y.shape)<=1 or pred_y.shape[1]==1:
            pred_y = pred_y > 0
        else:
            pred_y = pred_y.argmax(dim=1)
    return pred_y


def get_feature_name_map(hids,fids,latent_feature_names,input_feature_names):
    latent_input = pd.DataFrame(columns=['latent_feature_name','input_feature_name'])
    latent = np.array(latent_feature_names)[hids.numpy().astype(int)]
    raw = np.array(input_feature_names)[fids.numpy().astype(int)].squeeze()
    for i,(l, r) in enumerate(zip(latent,raw)):
        latent_input.loc[i,'latent_feature_name'] = l
        latent_input.loc[i,'input_feature_name'] = r
    return latent_input


def get_jacobian_feature_maps(int_j,shift,H=5,F=5):
    #print('in H F',H,F)
    importance = int_j
    dis_hids = torch.argsort(shift,descending=False)[:H]
    dis_importance = importance.reshape(int_j.shape[0],-1,int_j.shape[-1])[:,dis_hids,:]

    pos_fids = dis_importance.argsort(dim=2,descending=True)[...,:F]
    neg_fids = (-dis_importance).argsort(dim=2,descending=True)[...,:F]

    
    return dis_hids,pos_fids,neg_fids


def get_top_feature_for_hidden_state(hidden_input,latent_feature_names,input_feature_names,K=5):
    iname = ['top_input_feature_'+str(k) for k in range(K)]
    latent_input = pd.DataFrame(columns=['latent_feature']+iname)
    raw = np.array(input_feature_names)
    for i in range(hidden_input.shape[0]):
        latent_input.loc[i,'latent_feature'] = latent_feature_names[i]
        fids = (-hidden_input[i]).argsort()[:K]
        latent_input.loc[i,latent_input.columns[1:]] = raw[fids]

    return latent_input


def analyse_shift_by_corpus(model,times,all_sim,top_K,test_x,corpus_x,pred_corpus_y,corpus_y,
                            corpus_raw_ids,test_corpus_map,descending=True,
                            hidden_input_heatmap=None,H=5,F=5):
    
    pre_fix = 'top ' if descending else 'bottom'
    top_ids = all_sim.argsort(descending=descending)[:top_K]
    #print(top_ids)
    test_corpus_map[pre_fix + 'similarity'] = all_sim[top_ids]
    test_corpus_map[pre_fix + 'similar corpus pred labels'] = pred_corpus_y[top_ids]
    test_corpus_map[pre_fix + 'similar corpus true labels'] = corpus_y[top_ids]
    
    test_examples = corpus_x[top_ids]
        
    int_j,shift = integrad(test_examples=test_examples,model=model, 
                            input_baseline=test_x,n_bins=100,times=times)

    test_corpus_map[pre_fix + 'corpus'] = [corpus_raw_ids[t] for t in top_ids]#[get_corpus_raw_index(c,train_df) for c in top_ids]

    for c in range(len(top_ids)):
        dis_hids,pos_fids,neg_fids = get_jacobian_feature_maps(int_j[c].unsqueeze(0),shift[c].reshape(-1),H=H,F=F)
    
       ## top raw features that contributed most of the largest changes in hidden states 
        if hidden_input_heatmap is not None:
            for hid,fid in zip(dis_hids,pos_fids.squeeze()):
                hidden_input_heatmap[0][hid,fid] += 1.
            for hid,fid in zip(dis_hids,neg_fids.squeeze()):
                hidden_input_heatmap[1][hid,fid] += 1.
    return test_corpus_map,hidden_input_heatmap


class DataCorpus:
    def __init__(self,train_y,class_size=1000) -> None:
        self.class_size = class_size
        self.C = int(train_y.max())+1
        print('corpus classes',self.C)
        self.train_y = train_y
        
        self.class_idx = [self.sample_ids(c,train_y) for c in range(self.C)]
        
        
    def sample_ids(self,c,train_y):
        tot = (train_y==c).sum()
        replace = False if tot > self.class_size else True
        return np.random.choice(tot,size=self.class_size,replace=replace)
    
    def sample_data(self,train_x):
        cdata = []
        for c,cids in enumerate(self.class_idx):
            cdata.append(train_x[self.train_y==c][cids])
        if isinstance(train_x,torch.Tensor):
            return torch.cat(cdata,dim=0)
        elif isinstance(train_x,np.ndarray):
            return np.concatenate(cdata)
        else:
            raise TypeError('Not supported type!')
    
    def get_corpus_raw_index(self,train_df):
        raw_index = []
        for c,cids in enumerate(self.class_idx):
            raw_index += list(train_df.loc[train_df.label==c].index[cids]) 

        return raw_index




def explain_hidden_states_by_simplex(model,times,train_data,train_X_raw,train_df,
                                     test_data,test_X_raw,test_df,latent_feature_names,
                                     input_feature_names,model_kargs={'stream':True},
                                     top_K=10,corpus_class_size=1000,H=5,F=5):
    test_corpus_map ={}
    
    train_zt = data_to_hidden_states(train_data,model,times,stream=model_kargs['stream'],flat=False)
    test_zt = data_to_hidden_states(test_data,model,times,stream=model_kargs['stream'],flat=False)
    
    test_y = test_df.label.values
    train_y = train_df.label.values
    
    pred_train_y = data_to_prediction(train_data,model,times,stream=model_kargs['stream'],logits=False)
    pred_test_y = data_to_prediction(test_data,model,times,stream=model_kargs['stream'],logits=False)
    
    hidden_input_heatmap_pos = np.zeros([np.prod(train_zt.shape[1:]),train_X_raw.shape[-1]])
    hidden_input_heatmap_neg = np.zeros([np.prod(train_zt.shape[1:]),train_X_raw.shape[-1]])
    hidden_input_heatmap = (hidden_input_heatmap_pos,hidden_input_heatmap_neg)
    
    corpus = DataCorpus(train_y,class_size=corpus_class_size)
    corpus_zt = corpus.sample_data(train_zt)
    pred_corpus_y = corpus.sample_data(pred_train_y)
    corpus_y = corpus.sample_data(train_y)
    corpus_raw = corpus.sample_data(train_X_raw)
    corpus_raw_ids = corpus.get_corpus_raw_index(train_df)
    #print('corpus y',corpus_y.shape)
    for i in range(len(test_X_raw)):
        
        test_corpus_map[i] = {'test_raw_index':test_df.index[i]}

        test_corpus_map[i]['test true label'] = test_y[i]
        test_corpus_map[i]['test pred label'] = pred_test_y[i]

        all_sim = cosine_sim(corpus_zt,test_zt[i].unsqueeze(0),dims=[1,2])
        
        ## analyse top similar corpus
        test_corpus_map[i], _ = analyse_shift_by_corpus(model,times,all_sim,top_K=top_K,test_x=test_X_raw[i],
                                                                        corpus_x=corpus_raw,pred_corpus_y=pred_corpus_y,corpus_y=corpus_y,
                                                                        corpus_raw_ids=corpus_raw_ids,test_corpus_map=test_corpus_map[i],
                                                                        descending=True,hidden_input_heatmap=None)
        
        ## analyse bottom similar corpus
        test_corpus_map[i], hidden_input_heatmap = analyse_shift_by_corpus(model,times,all_sim,top_K=top_K,test_x=test_X_raw[i],
                                                                        corpus_x=corpus_raw,pred_corpus_y=pred_corpus_y,corpus_y=corpus_y,
                                                                        corpus_raw_ids=corpus_raw_ids,test_corpus_map=test_corpus_map[i],
                                                                        descending=False,hidden_input_heatmap=hidden_input_heatmap,H=H,F=F)
    

        print(i)

    return test_corpus_map,hidden_input_heatmap,corpus


def calc_baselines_intg(test_examples,model,baselines,times=None,C=2,target_dim=0,n_bins=100):
    int_g, zshift = [],[]
    cids = np.arange(C)
    # for k in cids:
    k = target_dim
    for kk in cids[cids!=k]:
        if times is None:
            int_g_k,latent_shift = integrad(test_examples=test_examples[k],model=model, 
                                                    input_baseline=baselines[kk],target_dim=target_dim,n_bins=n_bins)
        else:
            int_g_k,latent_shift = integrad(test_examples=test_examples[k],model=model, 
                                                    input_baseline=baselines[kk],target_dim=target_dim,n_bins=n_bins,times=times)
        int_g.append(int_g_k)
        zshift.append(latent_shift)
    if len(baselines) > C:
        tsamples = torch.vstack(test_examples)
        for kk in range(C,len(baselines)):
            if times is None:
                int_g_k,latent_shift = integrad(test_examples=tsamples,model=model, 
                                            input_baseline=baselines[kk],target_dim=target_dim,n_bins=n_bins)

            else:
                int_g_k,latent_shift = integrad(test_examples=tsamples,model=model, 
                                            input_baseline=baselines[kk],target_dim=target_dim,n_bins=n_bins,times=times)
            int_g.append(int_g_k)
            zshift.append(latent_shift)

    int_g = torch.vstack(int_g)
    zshift = torch.vstack(zshift)
    return int_g, zshift
   

def intg_score(int_g,latent_shift):
    return int_g/torch.abs(latent_shift)


def output_intg_score(latent_int_g,linear_weights,output_shift):

    y_int_g = torch.einsum("nthd,th->ntd",latent_int_g,linear_weights)
    y_int_g = y_int_g.reshape(y_int_g.shape[0],-1)/output_shift.reshape(output_shift.shape[0],-1)
    
    return y_int_g

def gen_intgrad_baselines(x,y,reps=None,zero_x=False,zero_z=False):
 
    C = int(y.max())+1
    baselines = []
    if isinstance(x,torch.Tensor):
        for c in range(C):            
            baselines.append(x[y==c].mean(dim=0).unsqueeze(0))
                    
        if zero_x:
            baselines.append(torch.zeros_like(x[0]).unsqueeze(0))
        if zero_z:
            reps_norm = torch.square(reps).sum(dim=[d for d in range(1,len(reps.shape))])
            bid = torch.argmin(reps_norm)
            baselines.append(x[bid])
        return torch.vstack(baselines)
    elif isinstance(x,np.ndarray):
        for c in range(C):
            baselines.append(x[y==c].mean(axis=0).reshape(1,-1))
        if zero_x:
            baselines.append(np.zeros_like(x[0]).reshape(1,-1))
        if zero_z:
            reps_norm = np.square(reps).sum(axis=[d for d in range(1,len(reps.shape))])
            bid = np.argmin(reps_norm)
            baselines.append(x[bid])
        return np.vstack(baselines)
    else:
        raise TypeError('Not supported type!')


def gen_balanced_subset(x,y,size_per_class=500,shuffle=False):
    C = int(y.max())+1
    subset = []
    for c in range(C):
        y_c = (y == c)
        if isinstance(y_c,torch.Tensor):
            y_c = y_c.numpy()
        if shuffle:
            id_c = np.random.choice(np.sum(y_c),size=size_per_class)
            subset.append(x[y_c][id_c])
        else:
            if size_per_class <= np.sum(y_c):
                subset.append(x[y_c][:size_per_class])
            else:
                id_c = np.arange(np.sum(y_c))
                id_rc = np.arange(size_per_class-np.sum(y_c))
                id_c = np.concatenate([id_c,id_rc])
                subset.append(x[y_c][id_c])
        
#     subset = torch.vstack(subset)
    return subset


def find_py_threshold(p_grids,pred_y_prob,true_y,c=1,high=True):
    max_s = 0.
    thd = 0.
    if high:
        for p in p_grids:
            tpr = (true_y[pred_y_prob>=p]==c).sum()/(pred_y_prob>=p).sum()
            cp = (true_y[pred_y_prob>=p]==c).sum()/(true_y==c).sum()
            f1 = (2*tpr*cp)/(tpr+cp)
            if f1 > max_s:
                thd = p
                max_s = f1
    else:
        for p in p_grids:
            tpr = (true_y[pred_y_prob<=p]==c).sum()/(pred_y_prob<=p).sum()
            cp = (true_y[pred_y_prob<=p]==c).sum()/(true_y==c).sum()
            f1 = (2*tpr*cp)/(tpr+cp)
            if f1 > max_s:
                thd = p
                max_s = f1
    return thd
