import torch
from model.options import HiDDenConfiguration
from model.discriminator import Discriminator

from model.encoder import Encoder
from model.resnet18 import ResNet


class Net:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device):
        super(Net, self).__init__()

        self.encoder = Encoder(configuration).to(device)
        self.decoder = ResNet(configuration).to(device)

        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc = torch.optim.Adam(self.encoder.parameters())
        self.optimizer_dec = torch.optim.Adam(self.decoder.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())