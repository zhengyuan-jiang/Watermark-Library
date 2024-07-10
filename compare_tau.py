import torch
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms

from regression import certify
import utils
from model import network


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser(description='Compute certified robustness of different smoothing methods.')
    parser.add_argument('--options-file', default='./checkpoint/options-and-config.pickle',
                        type=str, help='The file where the simulation options are stored')
    parser.add_argument('--checkpoint-file', default='./checkpoint/adversarial.pth',
                        type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', default=100, type=int, help='Testing batch size')
    parser.add_argument("--num-noise", default=100, type=int,
                        help="Number of noised images (To reproduce results in paper, use 10000)")
    parser.add_argument("--range", default=0.4, type=float, help="Range of perturbation")
    parser.add_argument("--sigma", default=0.1, type=float, help="Gaussian noise level")
    parser.add_argument("--alpha", default=0.001, type=float, help="Confidence")

    args = parser.parse_args()

    train_options, config, noise_config = utils.load_options(args.options_file)
    train_options.batch_size = args.batch_size

    checkpoint = torch.load(args.checkpoint_file)
    net = network.Net(config, device)
    utils.model_from_checkpoint(net, checkpoint)

    encoder = net.encoder
    decoder = net.decoder
    encoder.eval()
    decoder.eval()

    # False Positive Rate of Non-wateramrked images
    data_transforms = transforms.Compose([
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    validation_images = datasets.ImageFolder('./non_GenAI_dataset', data_transforms)
    val_data = torch.utils.data.DataLoader(validation_images, batch_size=train_options.batch_size,
                                           shuffle=False, num_workers=4)
    num_data = len(val_data) * args.batch_size

    watermarks = torch.from_numpy(np.load('./100watermarks.npy'))
    bitaccs = torch.empty((num_data, args.num_noise))
    batch_idx = 0

    for image, _ in tqdm(val_data):
        images = image.to(device)

        for j in range(args.num_noise):
            gaussian_noise = torch.randn(images.shape).to(device)
            noised_images = images + args.sigma * gaussian_noise
            decoded_watermarks = decoder(noised_images).detach().cpu()
            decoded_watermarks = (decoded_watermarks >= 0.5).int()

            bitacc = torch.sum(watermarks == decoded_watermarks, dim=1) / config.message_length
            bitaccs[batch_idx:batch_idx + args.batch_size, j] = bitacc

        batch_idx += args.batch_size

    bitaccs = bitaccs.numpy()
    certified_fp = np.zeros((int(args.range * 100), num_data))
    for index in range(int(args.range * 100)):
        perturbation = index * 0.01
        k_ = certify(perturbation, args)
        for l in range(num_data):
            bitacc_array = bitaccs[l]
            bitacc_array.sort()
            if k_ == -1:
                certified_fp[index, l] = 1
            else:
                certified_fp[index, l] = 1 - bitacc_array[k_]

    for tau in [0.73, 0.78, 0.83, 0.88, 0.93]:
        print('FPR when Detection threshold tau=', tau, ':',  np.sum(certified_fp >= tau, axis=1) / num_data)

    # False Negative Rate of Watermarked images
    val_data = utils.get_data_loaders_DB(config, train_options, dataset='large_random_1k', train=False)
    num_data = len(val_data) * args.batch_size

    watermarks = torch.from_numpy(np.load('./100watermarks.npy'))

    bitaccs = torch.empty((num_data, args.num_noise))
    batch_idx = 0

    for batch in tqdm(iter(val_data)):
        images = batch['image'].to(device)
        encoded_images = encoder(images, watermarks.to(device))

        for j in range(args.num_noise):
            gaussian_noise = torch.randn(encoded_images.shape).to(device)
            noised_images = encoded_images + args.sigma * gaussian_noise
            decoded_watermarks = decoder(noised_images).detach().cpu()
            decoded_watermarks = (decoded_watermarks >= 0.5).int()

            bitacc = torch.sum(watermarks == decoded_watermarks, dim=1) / config.message_length
            bitaccs[batch_idx:batch_idx + args.batch_size, j] = bitacc

        batch_idx += args.batch_size

    bitaccs = bitaccs.numpy()
    certified_fn = np.zeros((int(args.range * 100), num_data))
    for index in range(int(args.range * 100)):
        perturbation = index * 0.01
        k_ = certify(perturbation, args)
        for l in range(num_data):
            bitacc_array = bitaccs[l]
            bitacc_array.sort()
            if k_ == -1:
                continue
            else:
                certified_fn[index, l] = bitacc_array[k_]

    for tau in [0.73, 0.78, 0.83, 0.88, 0.93]:
        print('FNR when Detection threshold tau=', tau, ':', np.sum(certified_fn < tau, axis=1) / num_data)


if __name__ == '__main__':
    main()