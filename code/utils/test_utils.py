import pandas as pd
import numpy as np
import shap
import os

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix,precision_score,recall_score,accuracy_score,roc_auc_score

from imblearn.over_sampling import RandomOverSampler


def sensitivity_score(true_y,pred_y,cmx=None):
    if cmx is None:
        cmx = confusion_matrix(true_y,pred_y)
    return cmx[1,1]/cmx[1,:].sum()


def specificity_score(true_y,pred_y,cmx=None):
    if cmx is None:
        cmx = confusion_matrix(true_y,pred_y)
    return cmx[0,0]/cmx[0,:].sum()



def calc_score(sc_name,true_y,pred_y,cmx=None,**kargs):
    
    score_map = {'accuracy':accuracy_score,'f1':f1_score,'precision':precision_score,'recall':recall_score,
                 'sensitivity':sensitivity_score, 'specificity':specificity_score,'auc':roc_auc_score}
    
    sc_fn = score_map[sc_name]
    if sc_name in ['sensitivity','specificity']:
        sc = sc_fn(true_y,pred_y,cmx=cmx)
    else:
        sc = sc_fn(true_y,pred_y,**kargs)
    print(sc_name,sc)
    return sc


def calc_shap_values(clf,train_X,train_y,test_X,shap_values={}):
    if isinstance(clf,GradientBoostingClassifier):
        if max(train_y)>1:
            explainer = shap.Explainer(clf, model_output='probability', data=train_X)
        else:
            explainer = shap.TreeExplainer(clf, model_output='probability', data=train_X)
            
    elif isinstance(clf,LogisticRegression):
        explainer = shap.LinearExplainer(clf, model_output='probability', data=train_X,masker=train_X)

    else:
        explainer = shap.Explainer(clf, model_output='probability', data=train_X)

    shap_values[clf.__class__.__name__] = shap_values.get(clf.__class__.__name__,[])
    shap_values[clf.__class__.__name__].append(explainer.shap_values(test_X))
    
    return shap_values


def date_split_train_test(dfs,split,window_len=pd.Timedelta('7D'),index_level=1,granularity='D'):
    """
    Split train and test data by a split date

    Args:
        dfs (list or tuple): Pandas dataframes for spliting.
        split (DateTime): The split date.
        window_len (Timedelta): The max period of test data.
        index_level (int, optional): The index level of timestamps in a dataframe. Defaults to 1.
        granularity (str, optional): The granularity of time. Defaults to 'D'.

    Returns:
        lists: train and test dataframes
    """
    train_dfs,test_dfs = [],[]
    for df in dfs:
        train_x = df.loc[(pd.to_datetime(df.index.get_level_values(index_level))-pd.to_datetime(split))<=pd.Timedelta(0,granularity)]
        test_x = df.loc[((pd.to_datetime(df.index.get_level_values(index_level))-pd.to_datetime(split))>pd.Timedelta(0,granularity))
                               &((pd.to_datetime(df.index.get_level_values(index_level))-pd.to_datetime(split))<=window_len)]
        print('check date',np.sort(train_x.index.get_level_values(index_level))[-1],np.sort(test_x.index.get_level_values(index_level))[-1],np.sort(test_x.index.get_level_values(index_level))[0])
        test_dfs.append(test_x)
        train_dfs.append(train_x)
    return train_dfs,test_dfs


def slide_date_split_CV_sets(*dfs,K=10,window_len=pd.Timedelta('7D'),index_level=1,granularity='D',gen_valid_set=False):
    """
    Generate K-fold cross-validation sets by sliding time windows

    Args:
        dfs (list or tuple): Pandas dataframes for spliting.
        K (int, optional): K-fold cross-validation. Defaults to 10.
        window_len (Timedelta, optional): The max period of test data. Defaults to pd.Timedelta('7D').
        index_level (int, optional): The index level of timestamps in a dataframe. Defaults to 1.
        granularity (str, optional): The granularity of time. Defaults to 'D'.
        gen_valid_set (bool, optional): If generate validation set or not. Defaults to False.

    Returns:
        dist: K-fold sets
    """
    ## the dates of all df in dfs are assumed to be aligned
    dates = np.sort(dfs[0].loc[dfs[0].label==1].index.get_level_values(1))
    end_date = dates[-1]
    cv_sets = {}
    for k in range(K):    
        split = end_date - window_len
        train_dfs, test_dfs = date_split_train_test(dfs,split=split,window_len=window_len,
                                                    index_level=index_level,granularity=granularity)
        end_date = split
        if gen_valid_set:
            train_dfs, val_dfs = date_split_train_test(train_dfs,split=end_date - window_len,window_len=window_len,
                                                    index_level=index_level,granularity=granularity)
            cv_sets[k]=(train_dfs,test_dfs,val_dfs)
        else:
            cv_sets[k]=(train_dfs,test_dfs)
    return cv_sets



def slide_date_cross_validation(df,CLFs,window_len=pd.Timedelta('7D'),K=10,scores=['sensitivity','specificity','auc'],shap=[GradientBoostingClassifier],sampler=RandomOverSampler):
    # df: DataFrame, must inlcude 'date' and 'label' columns, CLFs are scikit-learn style
    label_scores = {}
    shap_values= {}
    
    tX, pY, prop = [],{},[]
    dates = np.sort(df.loc[df.label==1].index.get_level_values(1))
    end_date = dates[-1]

    for k in range(K):

        split = end_date - window_len

        train_X = df.loc[(pd.to_datetime(df.index.get_level_values(1))-pd.to_datetime(split))<=pd.Timedelta(0,'D')]
        test_X = df.loc[((pd.to_datetime(df.index.get_level_values(1))-pd.to_datetime(split))>pd.Timedelta(0,'D'))
                               &((pd.to_datetime(df.index.get_level_values(1))-pd.to_datetime(split))<=window_len)]
        print('check date',np.sort(train_X.index.get_level_values(1))[-1],np.sort(test_X.index.get_level_values(1))[-1],np.sort(test_X.index.get_level_values(1))[0])
        end_date = split
        tX.append(test_X)

        train_y = train_X.label.values
        test_y = test_X.label.values

        train_X = train_X.drop(['label'],axis=1).values
        test_X = test_X.drop(['label'],axis=1).values
        
        prop.append(test_y.sum()/test_y.shape[0])
        
        if sampler is not None:
            ros = sampler(random_state=k)
            train_X, train_y = ros.fit_resample(train_X, train_y)
            
        
        for clf in CLFs:
            print(clf)
            #scores = {}
            label_scores[clf.__class__.__name__]=label_scores.get(clf.__class__.__name__,{})


            clf.fit(train_X,train_y)
            pred_y = clf.predict(test_X)
            pY[clf] = pY.get(clf,[])
            pY[clf].append(clf.predict_proba(test_X))

            cmx = confusion_matrix(test_y,pred_y)
            print('confusion matrix on separate validation set',cmx)

            for sc_name in scores:
                label_scores[clf.__class__.__name__][sc_name] = label_scores[clf.__class__.__name__].get(sc_name,[])
                sc = calc_score(sc_name,test_y,pred_y,cmx=cmx)
                label_scores[clf.__class__.__name__][sc_name].append(sc)
        
            for C in shap:
                if isinstance(clf,C):
                    shap_values = calc_shap_values(clf,train_X,train_y,test_X,shap_values)
                    
    print('average proportion of positive samples over all validations',np.mean(prop))
    
    return label_scores,shap_values,tX,pY
        
        
def score_summary(raw_scores):
    re_table_dist,re_table_mean,re_table_std = {},{},{}

    for c in raw_scores.keys():
        print('#############################################')
        print(c)
        re_table_mean[c]={}
        re_table_std[c]={}
    
        for sc in raw_scores[c].keys():
            re_table_dist[sc] = re_table_dist.get(sc,{})
            re_table_dist[sc][c] = raw_scores[c][sc]

            print('{} mean {}, std {}'.format(sc,np.mean(raw_scores[c][sc]),np.std(raw_scores[c][sc])))
            
            re_table_mean[c][sc]= np.mean(np.mean(raw_scores[c][sc]))
            re_table_std[c][sc]= np.std(np.mean(raw_scores[c][sc]))
    return re_table_dist,re_table_mean,re_table_std


def make_results_filenames(args,dataset):
    args.rpath = os.path.join(args.rpath,dataset)
    os.makedirs(args.rpath,exist_ok=True)
    if args.intensity:
        name = 'intensity'
    if args.time_intensity:
        name += '_time_intensity'
    if args.concat_z:
        name += '_concatz'
    if args.side_input:
        name += '_sideinput'
    name += '/zdim'+str(args.hidden_channels)+'_hdim'+str(args.hidden_hidden_channels)+'_nlayer'+str(args.num_hidden_layers)+'_bs'+str(args.batch_size)
    name += '/posw'+str(args.pos_weight)
    name += '/interp_'+str(args.interpolate)
    return name