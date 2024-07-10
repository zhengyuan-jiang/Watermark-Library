import torch
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms

from smooth import multiclass, multilabel, regression
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
    parser.add_argument("--tau", default=0.83, type=float, help="Detection Threshold")
    parser.add_argument("--alpha", default=0.001, type=float, help="Confidence")
    parser.add_argument("--k", default=1, type=int, help="Top k parameter")

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

    watermark = 1 - torch.from_numpy(np.load('./watermark.npy')).to(device)
    save_watermarks = torch.empty((num_data, args.num_noise, config.message_length))
    batch_idx = 0


    for image, _ in tqdm(val_data):
        images = image.to(device)

        for i in range(args.num_noise):
            gaussian_noise = torch.randn(images.shape).to(device)
            noised_images = images + args.sigma * gaussian_noise
            decoded_watermarks = decoder(noised_images).detach().cpu()

            save_watermarks[batch_idx:batch_idx + args.batch_size, i, :] = decoded_watermarks

        batch_idx += args.batch_size

    save_watermarks = save_watermarks.numpy()
    watermark = watermark.detach().cpu().numpy()

    fp1 = np.empty([num_data, int(args.range*100)])
    fp2 = np.empty([num_data, int(args.range*100)])
    fp3 = np.empty([num_data, int(args.range*100)])

    for image_idx in tqdm(range(num_data)):
        fp1[image_idx, :] = 1 - multiclass(save_watermarks[image_idx, :, :], watermark, args, config)
        fp2[image_idx, :] = 1 - multilabel(save_watermarks[image_idx, :, :], watermark, args, config)
        fp3[image_idx, :] = 1 - regression(save_watermarks[image_idx, :, :], watermark, args, config)

    print("Multi-class FPR:", np.sum(fp1 >= args.tau, axis=0)/num_data)
    print("Multi-class FPR:", np.sum(fp2 >= args.tau, axis=0)/num_data)
    print("Regression FPR:", np.sum(fp3 >= args.tau, axis=0)/num_data)

    # False Negative Rate of Watermarked images
    val_data = utils.get_data_loaders_DB(config, train_options, dataset='large_random_1k', train=False)
    num_data = len(val_data) * args.batch_size

    watermark = torch.from_numpy(np.load('./watermark.npy')).to(device)
    save_watermarks = torch.empty((num_data, args.num_noise, config.message_length))
    batch_idx = 0

    for batch in tqdm(iter(val_data)):
        images = batch['image'].to(device)
        encoded_images = encoder(images, watermark.repeat(100, 1))

        for i in range(args.num_noise):
            gaussian_noise = torch.randn(images.shape).to(device)
            noised_images = encoded_images + args.sigma * gaussian_noise
            decoded_watermarks = decoder(noised_images).detach().cpu()

            save_watermarks[batch_idx:batch_idx + args.batch_size, i, :] = decoded_watermarks

        batch_idx += args.batch_size

    save_watermarks = save_watermarks.numpy()
    watermark = watermark.detach().cpu().numpy()

    fn1 = np.empty([num_data, int(args.range*100)])
    fn2 = np.empty([num_data, int(args.range*100)])
    fn3 = np.empty([num_data, int(args.range*100)])

    for image_idx in tqdm(range(num_data)):
        fn1[image_idx, :] = multiclass(save_watermarks[image_idx, :, :], watermark, args, config)
        fn2[image_idx, :] = multilabel(save_watermarks[image_idx, :, :], watermark, args, config)
        fn3[image_idx, :] = regression(save_watermarks[image_idx, :, :], watermark, args, config)

    print("Multi-class FNR:", np.sum(fn1 < args.tau, axis=0)/num_data)
    print("Multi-label FNR:", np.sum(fn2 < args.tau, axis=0)/num_data)
    print("Regression FNR:", np.sum(fn3 < args.tau, axis=0)/num_data)


if __name__ == '__main__':
    main()