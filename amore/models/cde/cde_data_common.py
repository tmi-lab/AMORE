import os

from sklearn.model_selection import train_test_split
import sys
import torch
import numpy as np

here = os.getcwd()
sys.path.append(os.path.join(here,"../"))

from cde_interp.interpolation_common import get_interp_coeffs

import controldiffeq

  
def dataloader(dataset, **kwargs):              
    if 'shuffle' not in kwargs:
        kwargs['shuffle'] = True
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = True
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 32
    kwargs['batch_size'] = min(kwargs['batch_size'], len(dataset))
    return torch.utils.data.DataLoader(dataset, **kwargs)

  
def get_final_indices(times,y,X=None,time_aligned=True):   
    if time_aligned:
        final_indices = np.ones_like(y,dtype=int)*(len(times)-1)    
        final_indices = torch.tensor(final_indices)
    else:
        final_indices = []
        for time in X:
            final_indices.append(len(time) - 1)
        maxlen = max(final_indices) + 1
        for time in X:
            for _ in range(maxlen - len(time)):
                time.append(np.zeros(len(time[0]))+np.nan)

        t = len(X[0]) 
        n = len(X)
        X = np.concatenate(X).reshape(n,t,-1)
        print('check X',X.shape)
        X = torch.tensor(X)
    return final_indices, X

def augment_data(X,times,intensity=True,time_intensity=True,append_times=True,cummean=False,cumsum=True):
    X_aug = []
    
    if time_intensity:
        if X.isnan().any():
            X_intensity = X.isnan().to(X.dtype).cumsum(dim=1)
        else:
            X_intensity = X!=0  # of size (batch, stream, channels)
            X_intensity = X_intensity.to(X.dtype).cumsum(dim=1)
        X_aug.append(X_intensity)
        
    
    if intensity:
        X_intensity = X.cummax(dim=1).values
        X_aug.append(X_intensity)
        if cumsum:
            X_intensity = X.cumsum(dim=1)
            X_aug.append(X_intensity)
        if cummean and time_intensity:
            X_intensity = X.cumsum(dim=1)/X_aug[0]
            X_intensity[X_intensity.isnan()] = 0
            X_aug.append(X_intensity)
        
        
    if append_times:   
        X_aug.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))
    if len(X_aug)>0:    
        X = torch.cat([X]+X_aug,dim=len(X.shape)-1)
    
    return X

def process_data(times,X,intensity=True,time_intensity=True,cummean=False,cumsum=True,append_times=True,interpolate='cubic_spline'):
    
    #final_indices,X = get_final_indices(times,y,X=X,time_aligned=time_aligned)    
    # X_aug = []

    # if intensity:
    #     X_intensity = X.cummax(dim=1).values
    #     X_aug.append(X_intensity)

    #     X_intensity = X.cumsum(dim=1)
    #     X_aug.append(X_intensity)

    # if time_intensity:
    #     X_intensity = X!=0  # of size (batch, stream, channels)
    #     X_intensity = X_intensity.to(X.dtype).cumsum(dim=1)
    #     X_aug.append(X_intensity)
    
    # if append_times:   
    #     X_aug.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))
    # if len(X_aug)>0:    
    #     X = torch.cat([X]+X_aug,dim=len(X.shape)-1)
    X = augment_data(X,times,intensity=intensity,time_intensity=time_intensity,cummean=cummean,cumsum=cumsum,append_times=append_times)
    print("check X",X.shape)
    coeffs = get_interp_coeffs(X=X,times=times,interpolate=interpolate,append_times=append_times)
        
    return coeffs

    

def pipeline(times,*dfs,transform_fn,side_input=False,intensity=False,time_intensity=False,append_times=True):
    X,y,X2 = transform_fn(*dfs)
    if not side_input:
        X2 = None
    coeffs,y = process_data(times,X,y,intensity=intensity,time_intensity=time_intensity,append_times=append_times)
    return coeffs,y,X2

def get_final_linear_input_channels(hidden_channels,side_input_dim=0,time_len=1):
       
    return hidden_channels * time_len+side_input_dim



def preprocess_data(times, X, y, final_index, append_times=True,side_input=None):
    #X = normalise_data(X, y)

    # Append extra channels together. Note that the order here: time, intensity, original, is important, and some models
    # depend on that order.
    augmented_X = []
    if append_times:
        augmented_X.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))
    augmented_X.append(X)
    if len(augmented_X) == 1:
        X = augmented_X[0]
    else:
        X = torch.cat(augmented_X, dim=2)

    print('X shape',X.shape,'y shape',y.shape)
    if side_input is None:
        data = [X,y,final_index]
    else:
        data = [X,y,final_index,side_input]
    
    train_val = train_test_split(*data,test_size=0.3,stratify=y)
    print('train val',len(train_val))
    train_data = [train_val[i] for i in range(0,len(train_val),2)]
    val_data = [train_val[i] for i in range(1,len(train_val),2)]
    
    val_test = train_test_split(*val_data,test_size=0.5,stratify=val_data[1])
    val_data = [val_test[i] for i in range(0,len(val_test),2)]
    test_data = [val_test[i] for i in range(1,len(val_test),2)]
         

    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, train_data[0])
    val_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, val_data[0])
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, test_data[0])

    train_data[0] = train_coeffs
    val_data[0] = val_coeffs
    test_data[0] = test_coeffs
    

    return times,train_data,val_data,test_data

class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.memory_dataset = {}

    def __getitem__(self, index):
        if index in self.memory_dataset:
            return self.memory_dataset[index]
        output = self.dataset[index]
        self.memory_dataset[index] = output
        return output

    def __len__(self):
        return len(self.dataset)
    

def wrap_dataloader(data,device,**kwargs):
    dlist = data_to_device(data,device)
    dataset = torch.utils.data.TensorDataset(*dlist)
    #if device!='cpu':
    dataset = MemoryDataset(dataset)
    dloader = dataloader(dataset, **kwargs)
    return dloader


def data_to_device(data,device):
    dlist = []
    for tr_data in data:
        if isinstance(tr_data,torch.Tensor):
            tr_data = tr_data.to(device)
            dlist.append(tr_data)
        if isinstance(tr_data,tuple):
            tr_data = [tdata.to(device) for tdata in tr_data]
            dlist += tr_data
    return dlist


def wrap_data(times, train_data, val_data, test_data, device, batch_size, num_workers=8,pin_memory=False):
    times = times.to(device)
    
    kwargs = {}
    kwargs['batch_size'] = batch_size
    if device=='cpu':
        kwargs['num_workers'] = num_workers
    if device=='cuda':
        kwargs['pin_memory'] = pin_memory
    
    train_dataloader = wrap_dataloader(train_data,device,**kwargs) 
    val_dataloader = wrap_dataloader(val_data,device,**kwargs) 
    test_dataloader = wrap_dataloader(test_data,device,**kwargs) 
    
    return times, train_dataloader, val_dataloader, test_dataloader

    

def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors



