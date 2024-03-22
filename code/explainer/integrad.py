# -----------------------------------------------------------------------------------------
# This work is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
#
# Author: Yu Chen
# Year: 2023
# Description: This file contains implementation of Integrated Gradients.
# -----------------------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn.functional as F


# function to extract grad
def set_grad(var):
    def hook(grad):
        print("grad shape",grad.shape,var.grad.shape )
        var.grad += grad
    return hook

class Var:
    def __init__(self,initial_value):
        self.grad = initial_value

def integrad(
        test_examples,
        model,
        input_baseline: torch.Tensor,
        n_bins: int = 100,
        target_dim: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        :param test_examples: the test examples for computing shifts
        :param model: the black-box model for which the Jacobians are computed
        :param input_baseline: the baseline input features
        :param n_bins: number of bins involved in the Riemann sum approximation for the integral
        :return:
        """
        test_inputs = test_examples.clone().requires_grad_()
        input_baseline = input_baseline.unsqueeze(0)

        input_shift = test_inputs - input_baseline
        baseline_rep = model.latent_representation(input_baseline,**kwargs).detach()
        test_latent_reps = model.latent_representation(test_examples,**kwargs).detach()
        latent_shift = test_latent_reps - baseline_rep 
        input_grad = 0.
        if hasattr(model, "customize_grad"):
            
            latent_grad = torch.ones_like(latent_shift)
            for n in range(1, n_bins + 1):
                t = n / n_bins
                input = input_baseline + t * (test_inputs - input_baseline)
                latent_reps = model.latent_representation(input,**kwargs)            
                input_grad += model.customize_grad(latent_reps, latent_grad).detach()
            
            ## input_shift is of shape (batch_size, time_steps, input_dim) ##
            ## input_grad is of shape (batch_size, time_steps, latent_dim, input_dim) ##
            integrated_grads = torch.unsqueeze(input_shift,2) * input_grad / (n_bins)
            integrated_grads[torch.isnan(integrated_grads)] = 0.
        else: 
            for n in range(1, n_bins + 1):
                t = n / n_bins
                input = input_baseline + t * (test_inputs - input_baseline)
                latent_reps = model.latent_representation(input,**kwargs)              
                latent_reps = latent_reps.reshape(latent_reps.shape[0],-1)
                e = latent_reps[:,target_dim]
                e.backward(gradient=torch.ones_like(e))               
                input_grad += test_inputs.grad
                test_inputs.grad.data.zero_()
                
            integrated_grads = input_shift * input_grad / n_bins

        return integrated_grads.detach(),latent_shift

