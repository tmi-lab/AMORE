import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F


 
    

def integrad(
        test_examples,
        model,
        input_baseline: torch.Tensor,
        n_bins: int = 100,
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
            integrated_grads = torch.einsum('bij,bijk->bijk',torch.abs(1./latent_shift),integrated_grads)
            integrated_grads[torch.isnan(integrated_grads)] = 0.
        else: 
            latent_shift_sqrdnorm = torch.sum(latent_shift**2, dim=-1, keepdim=True)
            for n in range(1, n_bins + 1):
                t = n / n_bins
                input = input_baseline + t * (test_inputs - input_baseline)
                latent_reps = model.latent_representation(input,**kwargs)              
                latent_reps.backward(gradient=latent_shift / latent_shift_sqrdnorm)
                input_grad += test_inputs.grad
                test_inputs.grad.data.zero_()
            integrated_grads = input_shift * input_grad / n_bins

        return integrated_grads.detach(),latent_shift

