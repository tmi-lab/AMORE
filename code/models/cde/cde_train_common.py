import copy
import json
import math
import numpy as np
import os

import sklearn.metrics
import torch
import tqdm

from . import FinalTanh,NeuralCDE,ODERNN,GRU_ODE,GRU_D,GRU_dt




def _add_weight_regularisation(loss_fn, regularise_parameters, scaling=0.03):
    def new_loss_fn(pred_y, true_y):
        total_loss = loss_fn(pred_y, true_y)
        for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                total_loss = total_loss + scaling * parameter.norm()
        return total_loss
    return new_loss_fn


class _SqueezeEnd(torch.nn.Module):
    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).squeeze(-1)


def _count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


class _AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]


def _evaluate_metrics(dataloader, model, times, loss_fn, num_classes, device, kwargs):
    with torch.no_grad():
        total_accuracy = 0
        total_confusion = np.zeros([num_classes, num_classes])#.numpy()  # occurs all too often
        total_dataset_size = 0
        total_loss = 0
        true_y_cpus = []
        pred_y_cpus = []

        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            if model.side_input:
                *coeffs, true_y, lengths, sinput = batch
                pred_y = model(times, coeffs, lengths,side_input=sinput, **kwargs)
            else:
                *coeffs, true_y, lengths = batch
                pred_y = model(times, coeffs, lengths, **kwargs)
            batch_size = true_y.size(0)
            
            true_y = true_y.detach().cpu()
            pred_y = pred_y.detach().cpu()
            if num_classes == 2:
                thresholded_y = (pred_y > 0).to(true_y.dtype)
            else:
                thresholded_y = torch.argmax(pred_y, dim=1)

            true_y_cpus.append(true_y)
            pred_y_cpus.append(pred_y)

            total_accuracy += (thresholded_y == true_y).sum().to(pred_y.dtype)
            total_confusion += sklearn.metrics.confusion_matrix(true_y, thresholded_y,
                                                                labels=range(num_classes))
            total_dataset_size += batch_size
            total_loss += loss_fn(pred_y, true_y) * batch_size

        
        total_loss /= total_dataset_size  # assume 'mean' reduction in the loss function
        total_accuracy /= total_dataset_size

        metrics = _AttrDict(accuracy=total_accuracy.item(), confusion=total_confusion, dataset_size=total_dataset_size,
                            loss=total_loss.item())

        if num_classes == 2:
            true_y_cpus = torch.cat(true_y_cpus, dim=0)
            pred_y_cpus = torch.cat(pred_y_cpus, dim=0)
            metrics.auroc = sklearn.metrics.roc_auc_score(true_y_cpus, pred_y_cpus)
            metrics.average_precision = sklearn.metrics.average_precision_score(true_y_cpus, pred_y_cpus)
        return metrics


class _SuppressAssertions:
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is AssertionError:
            self.tqdm_range.write('Caught AssertionError: ' + str(exc_val))
            return True


def _train_loop(train_dataloader, val_dataloader, model, times, optimizer, loss_fn, max_epochs, num_classes, device,
                kwargs, step_mode):
    model.train()
    best_model = model
    best_train_loss = math.inf
    best_val_accuracy = 0
    best_val_loss = math.inf

    save_auroc = 0
    save_train_loss = math.inf
    save_val_loss = math.inf
    best_train_loss_epoch = 0
    history = []
    breaking = False

    if step_mode:
        epoch_per_metric = 10
        plateau_terminate = 100
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    else:
        epoch_per_metric = 10
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, mode='max')

    tqdm_range = tqdm.tqdm(range(max_epochs))
    tqdm_range.write('Starting training for model:\n\n' + str(model) + '\n\n')
    for epoch in tqdm_range:
        if breaking:
            break
        for batch in train_dataloader:
            #batch = tuple(b.to(device) for b in batch)
            if breaking:
                break
            with _SuppressAssertions(tqdm_range):
                if model.side_input:
                    *train_coeffs, train_y, lengths, sinput = batch
                    pred_y = model(times, train_coeffs, lengths, side_input=sinput, **kwargs)
                else:
                    *train_coeffs, train_y, lengths = batch
                    pred_y = model(times, train_coeffs, lengths, **kwargs)
                loss = loss_fn(pred_y, train_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if epoch % epoch_per_metric == 0 or epoch == max_epochs - 1:
            model.eval()
            train_metrics = _evaluate_metrics(train_dataloader, model, times, loss_fn, num_classes, device, kwargs)
            val_metrics = _evaluate_metrics(val_dataloader, model, times, loss_fn, num_classes, device, kwargs)
            model.train()

            if train_metrics.loss < best_train_loss:
                best_train_loss = train_metrics.loss
                best_train_loss_epoch = epoch
                
            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss

            if val_metrics.accuracy > best_val_accuracy:
                best_val_accuracy = val_metrics.accuracy
                #best_train_accuracy_epoch = epoch

            if epoch>0 and val_metrics.loss < save_val_loss * 1.02 and train_metrics.loss < save_train_loss * 1.02:
                if not (val_metrics.loss>save_val_loss and val_metrics.auroc < save_auroc):
                    if val_metrics.auroc * 1.05 >= save_auroc:
                        del best_model  # so that we don't have three copies of a model simultaneously
                        best_model = copy.deepcopy(model)
                        best_epoch = epoch
                        save_auroc = val_metrics.auroc
                        save_val_loss = val_metrics.loss
                        save_train_loss = train_metrics.loss
                        print('save model')

            tqdm_range.write('Epoch: {}  Train loss: {:.3}  Train auroc: {:.3}  Val loss: {:.3}  '
                             'Val auroc: {:.3}'
                             ''.format(epoch, train_metrics.loss, train_metrics.auroc, val_metrics.loss,
                                       val_metrics.auroc))
            if step_mode:
                scheduler.step(train_metrics.loss)
            else:
                scheduler.step(val_metrics.accuracy)
            history.append(_AttrDict(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics))

            if epoch > best_train_loss_epoch + plateau_terminate:
                tqdm_range.write('Breaking because of no improvement in training loss for {} epochs.'
                                 ''.format(plateau_terminate))
                breaking = True
            # if epoch > best_train_accuracy_epoch + plateau_terminate:
            #     tqdm_range.write('Breaking because of no improvement in training accuracy for {} epochs.'
            #                      ''.format(plateau_terminate))
            #     breaking = True

    for parameter, best_parameter in zip(model.parameters(), best_model.parameters()):
        parameter.data = best_parameter.data
    print('best epoch',best_epoch)
    return history


class _TensorEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            super(_TensorEncoder, self).default(o)


def _save_results(name, result, path,model=None):
    loc = path
    
    loc = os.path.join(loc,name)  
    if not os.path.exists(loc):
        os.makedirs(loc,exist_ok=True)
    num = -1
    for filename in os.listdir(loc):
        try:
            num = max(num, int(filename))
        except ValueError:
            pass
    result_to_save = result.copy()
    del result_to_save['train_dataloader']
    del result_to_save['val_dataloader']
    del result_to_save['test_dataloader']
    result_to_save['model'] = str(result_to_save['model'])

    num += 1
    with open(os.path.join(loc, str(num)), 'w') as f:
        json.dump(result_to_save, f, cls=_TensorEncoder)
        
    if model is not None:
        torch.save(model.state_dict(), os.path.join(loc,'model_'+str(num)))
    
    return num


def main(name, times, train_dataloader, val_dataloader, test_dataloader, device, make_model, num_classes, max_epochs,
         lr, kwargs, step_mode, pos_weight=torch.tensor(1),rpath='./results'):
    times = times.to(device)
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None

    model, regularise_parameters = make_model()
    if num_classes == 2:
        #model = _SqueezeEnd(model)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.functional.cross_entropy
    loss_fn = _add_weight_regularisation(loss_fn, regularise_parameters)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = _train_loop(train_dataloader, val_dataloader, model, times, optimizer, loss_fn, max_epochs,
                          num_classes, device, kwargs, step_mode)

    model.eval()
    train_metrics = _evaluate_metrics(train_dataloader, model, times, loss_fn, num_classes, device, kwargs)
    val_metrics = _evaluate_metrics(val_dataloader, model, times, loss_fn, num_classes, device, kwargs)
    test_metrics = _evaluate_metrics(test_dataloader, model, times, loss_fn, num_classes, device, kwargs)

    if device != 'cpu':
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
    else:
        memory_usage = None

    result = _AttrDict(times=times,
                       memory_usage=memory_usage,
                       baseline_memory=baseline_memory,
                       num_classes=num_classes,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       test_dataloader=test_dataloader,
                       model=model.to('cpu'),
                       parameters=_count_parameters(model),
                       history=history,
                       train_metrics=train_metrics,
                       val_metrics=val_metrics,
                       test_metrics=test_metrics)
    
    sensitivity = result['test_metrics']['confusion'][1,1]/result['test_metrics']['confusion'][1,:].sum()
    specificity = result['test_metrics']['confusion'][0,0]/result['test_metrics']['confusion'][0,:].sum()
    result['test_metrics'].update({'sensitivity':sensitivity,'specificity':specificity})
    
    print('#####################')
    print('test_metrics')
    print(result['test_metrics'])
    print('#####################')
    print('val_metrics')
    print(result['val_metrics'])
    print('#####################')
    print('train_metrics')
    print(result['train_metrics'])
    
    num = 0
    if name is not None:
        num = _save_results(name, result,rpath,model=model.to('cpu'))
    return model,result,num


def make_model(name, input_channels, output_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
               use_intensity,final_linear_input_channels, initial=True,side_input=False,append_times=False,interpolate='cubic_spline'):
    if name == 'ncde':
        def make_model():
            vector_field = FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels,final_linear_input_channels=final_linear_input_channels, 
                                     initial=initial, side_input=side_input,interpolate=interpolate)
            return model, vector_field
    elif name == 'gruode':
        def make_model():
            vector_field = GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
            model = NeuralCDE(func=vector_field, input_channels=input_channels,
                                     hidden_channels=hidden_channels, output_channels=output_channels, initial=initial,interpolate=interpolate)
            return model, vector_field
    elif name == 'dt':
        def make_model():
            model = GRU_dt(input_channels=input_channels, hidden_channels=hidden_channels,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'decay':
        def make_model():
            model = GRU_D(input_channels=input_channels, hidden_channels=hidden_channels,
                                 output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'odernn':
        def make_model():
            model = ODERNN(input_channels=input_channels, hidden_channels=hidden_channels,
                                  hidden_hidden_channels=hidden_hidden_channels, num_hidden_layers=num_hidden_layers,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    else:
        raise ValueError("Unrecognised model name {}. Valid names are 'ncde', 'gruode', 'dt', 'decay' and 'odernn'."
                         "".format(name))
    return make_model
