#!/usr/bin/env python3

"""Actor critic."""
import pdb

import numpy as np
import os
import torch
import torch.nn as nn
import copy
import torchvision.transforms as T
try:
    import r3m
except:
    pass
import torchvision.models as models
from torch.distributions import MultivariateNormal
from utils.moco_load import UberModel, moco_conv3_compressed, moco_conv4_compressed, moco_conv5

from mvp.backbones import vit


###################################################################s############
# States
###############################################################################

class ActorCritic(nn.Module):

    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        encoder_cfg,
        policy_cfg
    ):
        super(ActorCritic, self).__init__()
        assert encoder_cfg is None

        actor_hidden_dim = policy_cfg['pi_hid_sizes']
        critic_hidden_dim = policy_cfg['vf_hid_sizes']
        activation = nn.SELU()

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(*obs_shape, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(*obs_shape, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    @torch.no_grad()
    def act(self, observations, states):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(observations)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
            None,  # dummy placeholder
        )

    @torch.no_grad()
    def act_inference(self, observations, states=None):
        actions_mean = self.actor(observations)
        return actions_mean

    def forward(self, observations, states, actions):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


###############################################################################
# Pixels
###############################################################################

_MODELS = {
    "vits-mae-hoi": "mae_pretrain_hoi_vit_small.pth",
    "vits-mae-in": "mae_pretrain_imagenet_vit_small.pth",
    "vits-sup-in": "sup_pretrain_imagenet_vit_small.pth",
    "vitb-mae-egosoup": "mae_pretrain_egosoup_vit_base.pth",
    "vitl-256-mae-egosoup": "mae_pretrain_egosoup_vit_large_256.pth",
    "moco":"/to/your/path/moco_aug.pth.tar",
    "r3m":" ",
}
_MODEL_FUNCS = {
    "vits": vit.vit_s16,
    "vitb": vit.vit_b16,
    "vitl": vit.vit_l16,
}


class Encoder(nn.Module):

    def __init__(self, model_name, pretrain_dir, freeze, emb_dim):
        super(Encoder, self).__init__()
        assert model_name in _MODELS, f"Unknown model name {model_name}"
        self.model_name = model_name
        self.transforms = nn.Sequential(
            T.Resize(256),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )
        if model_name == 'moco':
            print(' >>> Load moco model >>>>')

            self.backbone = moco_conv5(checkpoint_path=_MODELS[model_name])
            self.backbone.eval()

            x = torch.randn([1] + [3, 224, 224])
            with torch.no_grad():
                out = self.backbone(x)
            out = out.view(out.shape[0], -1)
            gap_dim = out.shape[1]

        elif model_name == 'r3m':
            self.backbone = r3m.load_r3m("resnet50")
            self.backbone.eval()
            x = torch.randn([1] + [3, 224, 224])

            with torch.no_grad():
                out = self.backbone.module.convnet(x)
            out = out.view(out.shape[0], -1)
            gap_dim = out.shape[1]

        else:
            model_func = _MODEL_FUNCS[model_name.split("-")[0]]
            img_size = 256 if "-256-" in model_name else 224
            pretrain_path = os.path.join(pretrain_dir, _MODELS[model_name])
            self.backbone, gap_dim = model_func(pretrain_path, img_size=img_size)
            if freeze:
                self.backbone.freeze()
            self.freeze = freeze
        self.projector = nn.Linear(gap_dim, emb_dim)

    @torch.no_grad()
    def forward(self, x):
        if self.model_name == 'moco' or self.model_name == 'moco_uber':
            x = x / 255.
            x = self.transforms(x)
            feat = self.backbone(x)
            return self.projector(feat), feat

        elif self.model_name =='r3m':
            x = x / 255.
            x = self.transforms(x)
            feat = self.backbone.module.convnet(x)
            return self.projector(feat), feat

        else:
            feat = self.backbone.extract_feat(x)
            return self.projector(self.backbone.forward_norm(feat)), feat

    def forward_feat(self, feat):
        if self.model_name == 'moco' or self.model_name == 'r3m':
            return self.projector(feat)
        else:
            return self.projector(self.backbone.forward_norm(feat))


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=3, emb_dim=384):
        super(SimpleEncoder, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.convnet = nn.Sequential(
            init_(nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )
        x = torch.randn([1] + [3, 224, 224])
        with torch.no_grad():
            out = self.convnet(x)
        out = out.view(out.shape[0], -1)
        self.projector = nn.Linear(out.shape[1], emb_dim)
        self.ln = nn.LayerNorm(out.shape[1])


    def forward(self, h):
        h = h / 255. - 0.5
        h = self.convnet(h)
        feat = h.view(h.shape[0], -1)
        return self.projector(self.ln(feat)), feat

    def forward_feat(self, feat):
        return self.projector(self.ln(feat))


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class TDMPC_Encoder(nn.Module):

    def __init__(self, in_channels=3, emb_dim=384):
        super(TDMPC_Encoder, self).__init__()

        num_channels = 32
        layers = [
                  nn.Conv2d(in_channels, num_channels, 7, stride=2), nn.ReLU(),
                  nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(),
                  nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(),
                  nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(),
                  nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(),
                  nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU()]
        self.convnet = nn.Sequential(*layers)
        x = torch.randn([1] + [3, 224, 224])
        with torch.no_grad():
            out = self.convnet(x)
        out = out.view(out.shape[0], -1)

        self.projector = nn.Linear(out.shape[1], emb_dim)
        self.ln = nn.LayerNorm(out.shape[1])

        self.apply(weight_init)


    def forward(self, h):
        h = h / 255. - 0.5
        h = self.convnet(h)
        feat = h.view(h.shape[0], -1)
        return self.projector(self.ln(feat)), feat

    def forward_feat(self, feat):
        return self.projector(self.ln(feat))




class PixelActorCritic(nn.Module):

    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        encoder_cfg,
        policy_cfg
    ):
        super(PixelActorCritic, self).__init__()
        assert encoder_cfg is not None

        # Encoder params
        model_type = encoder_cfg["model_type"]
        pretrain_dir = encoder_cfg["pretrain_dir"]
        pretrain_type = encoder_cfg["pretrain_type"]
        freeze = encoder_cfg["freeze"]
        emb_dim = encoder_cfg["emb_dim"]

        # Policy params
        actor_hidden_dim = policy_cfg["pi_hid_sizes"]
        critic_hidden_dim = policy_cfg["vf_hid_sizes"]
        activation = nn.SELU()

        # Encoder
        emb_dim = encoder_cfg["emb_dim"]

        self.obs_enc = Encoder(
            model_name=encoder_cfg["name"],
            pretrain_dir=encoder_cfg["pretrain_dir"],
            freeze=encoder_cfg["freeze"],
            emb_dim=emb_dim
        )
        self.state_enc = nn.Linear(states_shape[0], emb_dim)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(emb_dim * 2, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for li in range(len(actor_hidden_dim)):
            if li == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[li], *actions_shape))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dim[li], actor_hidden_dim[li + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(emb_dim * 2, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for li in range(len(critic_hidden_dim)):
            if li == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[li], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dim[li], critic_hidden_dim[li + 1])
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.obs_enc)
        print(self.state_enc)
        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    @torch.no_grad()
    def act(self, observations, states):
        obs_emb, obs_feat = self.obs_enc(observations)
        state_emb = self.state_enc(states)
        joint_emb = torch.cat([obs_emb, state_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(joint_emb)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
            obs_feat.detach(),  # return obs features
        )

    @torch.no_grad()
    def act_inference(self, observations, states):
        obs_emb, _ = self.obs_enc(observations)
        state_emb = self.state_enc(states)
        joint_emb = torch.cat([obs_emb, state_emb], dim=1)
        actions_mean = self.actor(joint_emb)
        return actions_mean

    def forward(self, obs_features, states, actions):
        obs_emb = self.obs_enc.forward_feat(obs_features)
        state_emb = self.state_enc(states)
        joint_emb = torch.cat([obs_emb, state_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.critic(joint_emb)

        return (
            actions_log_prob,
            entropy,
            value,
            actions_mean,
            self.log_std.repeat(actions_mean.shape[0], 1),
        )


class Simple_PixelActorCritic(PixelActorCritic):

    def __init__(self, obs_shape, states_shape, actions_shape,
                 initial_std, encoder_cfg, policy_cfg):
        super(Simple_PixelActorCritic, self).__init__(obs_shape, states_shape, actions_shape,
                                                      initial_std, encoder_cfg, policy_cfg)
        print('=========initialized with Simple_PixelActorCritic=============')

        del self.obs_enc
        # Obs and state encoders
        self.obs_enc = TDMPC_Encoder(
            in_channels=3,
            emb_dim=encoder_cfg["emb_dim"]
        )

        self.state_enc = nn.Linear(states_shape[0], 128)
        joint_emb_dim = encoder_cfg["emb_dim"] + 128

        emb_dim = encoder_cfg["emb_dim"]
        # Policy params
        actor_hidden_dim = policy_cfg["pi_hid_sizes"]
        critic_hidden_dim = policy_cfg["vf_hid_sizes"]
        activation = nn.ReLU()

        # Policy
        actor_layers = []

        actor_layers.append(nn.Linear(joint_emb_dim, joint_emb_dim))
        actor_layers.append(nn.LayerNorm(joint_emb_dim))
        actor_layers.append(nn.Tanh())

        actor_layers.append(nn.Linear(joint_emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for li in range(len(actor_hidden_dim)):
            if li == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[li], *actions_shape))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dim[li], actor_hidden_dim[li + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        # added trunk
        critic_layers.append(nn.Linear(joint_emb_dim, joint_emb_dim))
        critic_layers.append(nn.LayerNorm(joint_emb_dim))
        critic_layers.append(nn.Tanh())

        critic_layers.append(nn.Linear(joint_emb_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for li in range(len(critic_hidden_dim)):
            if li == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[li], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dim[li], critic_hidden_dim[li + 1])
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        self.apply(weight_init)

        print(self.obs_enc)
        print(self.state_enc)
        print(self.actor)
        print(self.critic)
