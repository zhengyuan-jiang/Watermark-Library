import os
import pickle
from datasets import load_dataset
from torchvision import datasets, transforms
import torch

import sys
sys.path.append("./model/")


def get_data_loaders_DB(config, train_options, dataset: str, train: bool):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((config.H, config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((config.H, config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    data = load_dataset('poloclub/diffusiondb', dataset, split='train')
    if train:
        images = data.map(lambda item: {'image': data_transforms['train'](item['image'])})
    else:
        images = data.map(lambda item: {'image': data_transforms['test'](item['image'])})

    images.set_format(type='torch', columns=['image', 'prompt'])

    data_loader = torch.utils.data.DataLoader(images, batch_size=train_options.batch_size,
                                              shuffle=False, num_workers=4)
    return data_loader


def load_options(options_file_name):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        noise_config = pickle.load(f)
        config = pickle.load(f)
        # for backward-capability. Some models were trained and saved before .enable_fp16 was added
        if not hasattr(config, 'enable_fp16'):
            setattr(config, 'enable_fp16', False)

    return train_options, config, noise_config


def model_from_checkpoint(net, checkpoint):
    net.encoder.load_state_dict(checkpoint['enc-model'])
    net.decoder.load_state_dict(checkpoint['dec-model'])
    net.optimizer_enc.load_state_dict(checkpoint['enc-optim'])
    net.optimizer_dec.load_state_dict(checkpoint['dec-optim'])
    net.discriminator.load_state_dict(checkpoint['discrim-model'])
    net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])