import torch
import torch.nn as nn
import yaml
import argparse

from model9_NS_transformer.diffusion_models.diffusion_utils import *
from model9_NS_transformer.diffusion_models.residual_patch_denoiser import ResidualPatchDenoiser


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class Model(nn.Module):
    """
    TMDM-r diffusion model with SimDiff-style residual patch denoiser.

    This version is the first clean ablation:
        - r_prior remains zero in exp_main.py
        - denoiser is replaced by ResidualPatchDenoiser
        - training loss remains pure noise prediction loss

    Forward interface is kept compatible with exp_main.py:

        output = self.model(batch_x, batch_x_mark, y_base, r_t_batch, r_prior, t)

    Inputs:
        x:        historical input, [B, seq_len, C]
        x_mark:   time features, kept for compatibility, unused here
        y_base:   base forecast from NS-Transformer, [B, pred_len, C]
        r_t:      noisy residual, [B, pred_len, C]
        r_prior:  residual prior center, currently unused, expected zero
        t:        diffusion step, [B]

    Output:
        eps:      predicted noise, [B, pred_len, C]
    """

    def __init__(self, configs, device):
        super(Model, self).__init__()

        with open(configs.diffusion_config_dir, "r") as f:
            config = yaml.unsafe_load(f)
            diffusion_config = dict2namespace(config)

        diffusion_config.diffusion.timesteps = configs.timesteps

        self.args = configs
        self.device = device
        self.diffusion_config = diffusion_config

        self.model_var_type = diffusion_config.model.var_type
        self.num_timesteps = diffusion_config.diffusion.timesteps
        self.vis_step = diffusion_config.diffusion.vis_step
        self.num_figs = diffusion_config.diffusion.num_figs
        self.dataset_object = None

        betas = make_beta_schedule(
            schedule=diffusion_config.diffusion.beta_schedule,
            num_timesteps=self.num_timesteps,
            start=diffusion_config.diffusion.beta_start,
            end=diffusion_config.diffusion.beta_end,
        )

        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)

        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)

        alphas_cumprod = alphas.to("cpu").cumprod(dim=0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

        if diffusion_config.diffusion.beta_schedule == "cosine":
            # avoid division by zero for 1 / sqrt(alpha_bar_t) during inference
            self.one_minus_alphas_bar_sqrt *= 0.9999

        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), alphas_cumprod[:-1]],
            dim=0,
        )

        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.posterior_mean_coeff_1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.posterior_mean_coeff_2 = (
            torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.posterior_variance = posterior_variance

        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.tau = None

        # Replace the original MLP denoiser with a SimDiff-style patch Transformer denoiser.
        self.denoiser = ResidualPatchDenoiser(configs, self.num_timesteps)

    def forward(self, x, x_mark, y_base, r_t, r_prior, t):
        """
        Predict diffusion noise for residual r_t.

        r_prior is intentionally unused in this first ablation:
            r_prior = 0
            Residual Patch Transformer
            noise loss
        """
        eps = self.denoiser(x, y_base, r_t, t)
        return eps
