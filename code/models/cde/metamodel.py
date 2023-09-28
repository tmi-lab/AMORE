
import sys
import torch


import controldiffeq

from models.cde_interp import LinearInterpolation,get_interp_coeffs




class NeuralCDE(torch.nn.Module):
    """A Neural CDE model. Provides a wrapper around the lower-level cdeint function, to get a flexible Neural CDE
    model.

    Specifically, considering the CDE
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where X is determined by the data, and given some terminal time t_N, then this model first computes z_{t_N}, then
    performs a linear function on it, and then outputs the result.

    It's known that linear functions on CDEs are universal approximators, so this is a very general type of model.
    """
    def __init__(self, func, input_channels, hidden_channels, output_channels,final_linear_input_channels=None, 
                 initial=True,side_input=False,append_times=True,interpolate='cubic_spline'):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        if isinstance(func, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(NeuralCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.final_linear_input_channles = final_linear_input_channels
        self.side_input = side_input
        self.append_times = append_times
        self.interpolate = interpolate
        print('interpolate',self.interpolate)

        self.func = func
        self.initial = initial
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        if final_linear_input_channels == None:            
            self.linear = torch.nn.Linear(hidden_channels, output_channels)
        else:
            self.linear = torch.nn.Linear(final_linear_input_channels, output_channels)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def latent_representation(self,X,times):
        X = X.reshape(X.shape[0],times.shape[0],-1)
        if self.append_times:
            atime = times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1)
            X = torch.cat([X,atime],dim=len(X.shape)-1)
        with torch.no_grad():            
            coeffs = get_interp_coeffs(X=X,times=times,interpolate=self.interpolate,append_times=self.append_times)
            #print('coeffs',len(coeffs))
            #controldiffeq.natural_cubic_spline_coeffs(times, X)
            Z = self.hidden_state(times,coeffs,stream=True,flat=False)  
         
        return Z 
    
    def customize_grad(self,Z,z_grad):
        if self.append_times:
            return self.func(Z)[...,:-1]
        else:
            return self.func(Z)
    
    
    
    def hidden_state(self, times, coeffs, final_index=None, z0=None, stream=False, flat=True, **kwargs):
        # Extract the sizes of the batch dimensions from the coefficients
        if isinstance(coeffs,torch.Tensor):
            batch_dims = coeffs.shape[:-2]
            data_type = coeffs.dtype
        else:
            coeff = coeffs[0]
            batch_dims = coeff.shape[:-2]
            data_type = coeff.dtype

        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        if self.interpolate == 'cubic_spline':
            interp_func = controldiffeq.NaturalCubicSpline(times, coeffs)
        elif self.interpolate == 'linear':
            interp_func = LinearInterpolation(coeffs, t=times)
        elif self.interpolate == 'rectilinear':
            interp_func = LinearInterpolation(coeffs, t=None)
        else:
            raise TypeError('Not supported interpolation type!')

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=data_type, device=coeff.device)
            else:
                z0 = self.initial_network(interp_func.evaluate(times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # continuing adventures in ugly hacks
                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)

        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat([times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # Actually solve the CDE
        z_t = controldiffeq.cdeint(dX_dt=interp_func.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=t,
                                   **kwargs)

        # Organise the output
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            if self.final_linear_input_channles is None:
                for i in range(len(z_t.shape) - 2, 0, -1):
                    z_t = z_t.transpose(0, i)
            else:
                z_t = z_t.transpose(0,1)
                if flat:
                    z_t = z_t.reshape(z_t.shape[0],-1)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)
        
        return z_t


    def forward(self, times, coeffs, final_index,side_input=None, z0=None, stream=False, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """
        z_t = self.hidden_state(times=times,coeffs=coeffs,final_index=final_index,
                                z0=z0,stream=stream,**kwargs)
        # Linear map and return
        if side_input is None:
            pred_y = self.linear(z_t)
        else:
            pred_y = self.linear(torch.cat((z_t,side_input),dim=1))
        if self.output_channels==1:
            pred_y = pred_y.squeeze(-1)
        return pred_y


# Note that this relies on the first channel being time
class ContinuousRNNConverter(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousRNNConverter, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model

        out_base = torch.zeros(self.input_channels + self.hidden_channels, self.input_channels)
        for i in range(self.input_channels):
            out_base[i, i] = 1
        self.register_buffer('out_base', out_base)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(self.input_channels, self.hidden_channels)

    def forward(self, z):
        # z is a tensor of shape (..., input_channels + hidden_channels)
        x = z[..., :self.input_channels]
        h = z[..., self.input_channels:]
        # In theory the hidden state must lie in this region. And most of the time it does anyway! Very occasionally
        # it escapes this and breaks everything, though. (Even when using adaptive solvers or small step sizes.) Which
        # is kind of surprising given how similar the GRU-ODE is to a standard negative exponential problem, we'd
        # expect to get absolute stability without too much difficulty. Maybe there's a bug in the implementation
        # somewhere, but not that I've been able to find... (and h does only escape this region quite rarely.)
        h = h.clamp(-1, 1)
        # model_out is a tensor of shape (..., hidden_channels)
        model_out = self.model(x, h)
        batch_dims = model_out.shape[:-1]
        out = self.out_base.repeat(*batch_dims, 1, 1).clone()
        out[..., self.input_channels:, 0] = model_out
        return out
