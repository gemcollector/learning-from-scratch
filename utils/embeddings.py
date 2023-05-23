import numpy as np
import gym
from gym.spaces.box import Box
import os
import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision.models as models
import torchvision.transforms as T
from utils.moco_load import moco_conv5, moco_conv3_compressed, moco_conv4_compressed


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# ==============================================================================
# GET EMBEDDING
# ==============================================================================

class UberModel(nn.Module):
    def __init__(self, models):
        super(UberModel, self).__init__()
        self.models = models
        assert all(models[0].training == m.training for m in models)
        self.training = models[0].training

    def to(self, device):
        self.models = [m.to(device=device) for m in self.models]
        return self

    def forward(self, x):
        return torch.cat([m(x) for m in self.models],
            dim=1 if x.ndim > 1 else 0)


def _get_embedding(embedding_name='random', in_channels=3, pretrained=True, train=False):
    """
    See https://pytorch.org/vision/stable/models.html
    Args:
        embedding_name (str, 'random'): the name of the convolution model,
        in_channels (int, 3): number of channels of the input image,
        pretrained (bool, True): if True, the model's weights will be downloaded
            from torchvision (if possible),
        train (bool, False): if True the model will be trained during learning,
            if False its parameters will not change.
    """

    # Default transforms: https://pytorch.org/vision/stable/models.html
    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    # where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then
    # normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    transforms = nn.Sequential(
        T.Resize(256, interpolation=3) if 'mae' in embedding_name else T.Resize(256),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )

    assert in_channels == 3, 'Current models accept 3-channel inputs only.'

    # FIXED 5-LAYER CONV
    if embedding_name == 'random':
        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        # original model
        model = nn.Sequential(
            init_(nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
        )
    elif embedding_name == 'ours':

        # model = nn.Sequential(
        #     init_(nn.Conv2d(in_channels, 32, kernel_size=(5,5), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        # )
        model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=1),
            # nn.ReLU(),
        )




        # # DrQ like model
        # model = nn.Sequential(
        #     init_(nn.Conv2d(in_channels, 32, kernel_size=(7,7), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(5,5), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        # )
        # remove imgnet normalization and resize to 84x84
        transforms = nn.Sequential(
            T.Resize(84),
        )


    elif embedding_name == 'mvp':
        import mvp
        # model = mvp.load("vitb-mae-egosoup", ckpt_path="checkpoints/vitb-mae-egosoup.pt")
        model = mvp.load("vits-mae-hoi", ckpt_path="checkpoints/vits-mae-hoi.pth")
        model.freeze()

    elif embedding_name == 'r3m':
        from r3m import load_r3m
        model = load_r3m("resnet50") # resnet18, resnet34
        model.eval()
    # Make FC layers to be Identity
    # This works for the models below but may not work for any network


    # MOCO
    elif embedding_name == 'moco_aug':
        model = moco_conv5(checkpoint_path='/home/yzc/pretrained_model/pvr_moco/moco_aug.pth.tar')
    elif embedding_name == 'moco_aug_habitat':
        model = moco_conv5(checkpoint_path='checkpoints/moco_aug_habitat_64.pth')
    elif embedding_name == 'moco_aug_mujoco':
        model = moco_conv5(checkpoint_path='checkpoints/moco_aug_mujoco.pth')
    elif embedding_name == 'moco_aug_uber':
        model = moco_conv5(checkpoint_path='checkpoints/moco_aug_uber.pth')
    elif embedding_name == 'moco_aug_places':
        model = moco_conv5(checkpoint_path='checkpoints/moco_aug_places.pth.tar')

    elif embedding_name == 'moco_aug_l4':
        model = moco_conv4_compressed(checkpoint_path='/home/yzc/pretrained_model/pvr_moco/moco_aug_l4.pth')
    elif embedding_name == 'moco_aug_places_l4':
        model = moco_conv4_compressed(checkpoint_path='/home/yzc/pretrained_model/pvr_moco/moco_aug_places_l4.pth')
    elif embedding_name == 'moco_aug_l3':
        model = moco_conv3_compressed(checkpoint_path='/home/yzc/pretrained_model/pvr_moco/moco_aug_l3.pth')
    elif embedding_name == 'moco_aug_places_l3':
        model = moco_conv3_compressed(checkpoint_path='checkpoints/moco_aug_places_l3.pth')

    elif embedding_name == 'moco_croponly':
        model = moco_conv5(checkpoint_path='checkpoints/moco_croponly.pth')
    elif embedding_name == 'moco_croponly_places':
        model = moco_conv5(checkpoint_path='checkpoints/moco_croponly_places.pth')
    elif embedding_name == 'moco_croponly_habitat':
        model = moco_conv5(checkpoint_path='checkpoints/moco_croponly_habitat_64.pth')
    elif embedding_name == 'moco_croponly_mujoco':
        model = moco_conv5(checkpoint_path='checkpoints/moco_croponly_mujoco.pth')
    elif embedding_name == 'moco_croponly_uber':
        model = moco_conv5(checkpoint_path='checkpoints/moco_croponly_uber.pth')

    elif embedding_name == 'moco_croponly_l4':
        model = moco_conv4_compressed(checkpoint_path='checkpoints/moco_croponly_l4.pth')
    elif embedding_name == 'moco_croponly_l3':
        model = moco_conv3_compressed(checkpoint_path='checkpoints/moco_croponly_l3.pth')
    elif embedding_name == 'moco_croponly_places_l4':
        model = moco_conv4_compressed(checkpoint_path='checkpoints/moco_croponly_places_l4.pth')
    elif embedding_name == 'moco_croponly_places_l3':
        model = moco_conv3_compressed(checkpoint_path='checkpoints/moco_croponly_places_l3.pth')

    elif embedding_name == 'moco_coloronly':
        model = moco_conv5(checkpoint_path='checkpoints/moco_coloronly.pth')

    # MOCO UBER MODELS (AUG)
    elif embedding_name == 'moco_aug_places_uber_345':
        model = UberModel([
            _get_embedding('moco_aug_places_l3')[0],
            _get_embedding('moco_aug_places_l4')[0],
            _get_embedding('moco_aug_places')[0]
        ])
    elif embedding_name in ['moco_aug_uber_345', 'moco']:
        model = UberModel([
            _get_embedding('moco_aug_l3')[0],
            _get_embedding('moco_aug_l4')[0],
            _get_embedding('moco_aug')[0]
        ])
    elif embedding_name == 'moco_aug_places_uber_35':
        model = UberModel([
            _get_embedding('moco_aug_places_l3')[0],
            _get_embedding('moco_aug_places')[0]
        ])
    elif embedding_name == 'moco_aug_uber_35':
        model = UberModel([
            _get_embedding('moco_aug_l3')[0],
            _get_embedding('moco_aug')[0]
        ])
    elif embedding_name == 'moco_aug_places_uber_34':
        model = UberModel([
            _get_embedding('moco_aug_places_l3')[0],
            _get_embedding('moco_aug_places_l4')[0],
        ])
    elif embedding_name == 'moco_aug_uber_34':
        model = UberModel([
            _get_embedding('moco_aug_l3')[0],
            _get_embedding('moco_aug_l4')[0],
        ])
    elif embedding_name == 'moco_aug_places_uber_45':
        model = UberModel([
            _get_embedding('moco_aug_places_l4')[0],
            _get_embedding('moco_aug_places')[0]
        ])
    elif embedding_name == 'moco_aug_uber_45':
        model = UberModel([
            _get_embedding('moco_aug_l4')[0],
            _get_embedding('moco_aug')[0]
        ])

    # MOCO UBER MODELS (CROP)
    elif embedding_name == 'moco_croponly_places_uber_345':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3')[0],
            _get_embedding('moco_croponly_places_l4')[0],
            _get_embedding('moco_croponly_places')[0]
        ])
    elif embedding_name == 'moco_croponly_uber_345':
        model = UberModel([
            _get_embedding('moco_croponly_l3')[0],
            _get_embedding('moco_croponly_l4')[0],
            _get_embedding('moco_croponly')[0]
        ])
    elif embedding_name == 'moco_croponly_places_uber_35':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3')[0],
            _get_embedding('moco_croponly_places')[0]
        ])
    elif embedding_name == 'moco_croponly_uber_35':
        model = UberModel([
            _get_embedding('moco_croponly_l3')[0],
            _get_embedding('moco_croponly')[0]
        ])
    elif embedding_name == 'moco_croponly_places_uber_34':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3')[0],
            _get_embedding('moco_croponly_places_l4')[0],
        ])
    elif embedding_name == 'moco_croponly_uber_34':
        model = UberModel([
            _get_embedding('moco_croponly_l3')[0],
            _get_embedding('moco_croponly_l4')[0],
        ])
    elif embedding_name == 'moco_croponly_places_uber_45':
        model = UberModel([
            _get_embedding('moco_croponly_places_l4')[0],
            _get_embedding('moco_croponly_places')[0]
        ])
    elif embedding_name == 'moco_croponly_uber_45':
        model = UberModel([
            _get_embedding('moco_croponly_l4')[0],
            _get_embedding('moco_croponly')[0]
        ])


    # TRUE STATE (BASELINE)
    elif embedding_name == 'true_state':
        return nn.Sequential(Identity()), nn.Sequential(Identity())

    else:
        raise NotImplementedError("Requested model not available.")

    if train:
        model.train()
        for p in model.parameters():
            p.requires_grad = True
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model, transforms


# ==============================================================================
# EMBEDDING CLASS
# ==============================================================================

class EmbeddingNet(nn.Module):
    """
    Input shape must be (N, H, W, 3), where N is the number of frames.
    The class will then take care of transforming and normalizing frames.
    The output shape will be (N, O), where O is the embedding size.
    """
    def __init__(self, embedding_name, in_channels=3, pretrained=True, train=False, disable_cuda=False):
        super(EmbeddingNet, self).__init__()

#        assert not train, 'Training the embedding is not supported.'

        self.embedding_name = embedding_name

        if self.embedding_name == 'true_state':
            return

        self.in_channels = in_channels
        self.embedding, self.transforms = \
            _get_embedding(embedding_name, in_channels, pretrained, train)
        dummy_in = torch.zeros(1, in_channels, 224, 224)
        dummy_in = self.transforms(dummy_in)
        self.in_shape = dummy_in.shape[1:]
        dummy_out = self._forward(dummy_in)
        self.out_size = np.prod(dummy_out.shape)

        # Always use CUDA, it is much faster for these models
        # Disable it only for debugging
        if torch.cuda.is_available() and not disable_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.embedding = self.embedding.to(device=self.device)
        self.training = self.embedding.training

    def _forward(self, observation):
        if 'clip' in self.embedding_name:
            out = self.embedding.encode_image(observation)
        elif 'mae' in self.embedding_name:
            out, *_ = self.embedding.forward_encoder(observation, mask_ratio=0.0)
            out = out[:,0,:]
        else:
            out = self.embedding(observation)
            if self.embedding_name == 'maskrcnn_l3':
                out = out['res4']
        return out

    def forward(self, observation):
        if self.embedding_name == 'true_state':
            return observation.squeeze().cpu().numpy()
        # observation.shape -> (N, H, W, 3)
        observation = observation.to(device=self.device)
        observation = observation.transpose(1, 2).transpose(1, 3).contiguous()
        observation = self.transforms(observation)

        if self.embedding.training:
            out = self._forward(observation)
            return out.view(-1, self.out_size).squeeze()
        else:
            with torch.no_grad():
                out = self._forward(observation)
                return out.view(-1, self.out_size).squeeze().cpu().numpy()




