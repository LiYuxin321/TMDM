import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
import yaml
import argparse
from model9_NS_transformer.diffusion_models.diffusion_utils import *
from model9_NS_transformer.diffusion_models.model import ConditionalGuidedModel



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
    Vanilla Transformer
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

        betas = make_beta_schedule(schedule=diffusion_config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=diffusion_config.diffusion.beta_start, end=diffusion_config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if diffusion_config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0
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
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.tau = None  # precision fo test NLL computation

        # CATE MLP
        self.diffussion_model = ConditionalGuidedModel(diffusion_config, self.args)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.CART_input_x_embed_dim, configs.embed, configs.freq,
                                           configs.dropout)

        a = 0

    def forward(self, x, x_mark, y, y_t, y_0_hat, t):
        enc_out = self.enc_embedding(x, x_mark)
        dec_out = self.diffussion_model(enc_out, y_t, y_0_hat, t)

        return dec_out
