import torch
import numpy as np

from multiclass import compute_r
from multilabel import Get_Overlap
from regression import certify


def multiclass(save_watermarks, watermark, args, config):
    # Use Multi-class based smoothing
    save_watermarks = (save_watermarks >= 0.5)
    flipped_watermarks = 1 - watermark
    save_watermarks = save_watermarks.sum(axis=0)
    count_vector = np.abs(save_watermarks - flipped_watermarks * args.num_noise)

    count_y = count_vector.reshape((config.message_length, 1))
    count_n = np.full((config.message_length, 1), args.num_noise) - count_y
    classes = np.zeros((config.message_length, 1), dtype=int)
    cv = np.concatenate((count_y, count_n, classes), axis=1)
    cr_array = compute_r(cv, args)
    cr_array = np.sort(cr_array)[::-1]

    perturbation_array = np.linspace(0, args.range, int(args.range*100))
    certified_ba_array = np.zeros(int(args.range*100))
    for idx in range(int(args.range*100)):
        certified_ba_array[idx] = np.sum(cr_array > perturbation_array[idx]) / config.message_length

    return certified_ba_array


def multilabel(save_watermarks, watermark, args, config):
    # Use Multi-label based smoothing
    save_watermarks = torch.from_numpy(save_watermarks)
    watermark = watermark.reshape((config.message_length))

    # ground-truth ones
    gt_ones_index = np.where(watermark == 1)[0]
    k_prime1 = len(gt_ones_index)  # number of predicted ones
    _, pred_ones_index = torch.topk(save_watermarks, k_prime1)

    count_vector1 = np.zeros(config.message_length, dtype=int)
    for j in range(config.message_length):
        count_vector1[j] = np.sum(pred_ones_index.numpy() == j)

    # ground-truth zeros
    gt_zeros_index = np.where(watermark == 0)[0]
    k_prime0 = len(gt_zeros_index)
    _, pred_zeros_index = torch.topk(-save_watermarks, k_prime0)

    count_vector0 = np.zeros(config.message_length, dtype=int)
    for j in range(config.message_length):
        count_vector0[j] = np.sum(pred_zeros_index.numpy() == j)

    perturbation_array = np.linspace(0, args.range, int(args.range*100))
    certified_ba_array = np.zeros(int(args.range*100))
    for idx in range(int(args.range*100)):
        e1 = Get_Overlap(count_vector1, watermark, k_prime1, k_prime1, args.alpha, perturbation_array[idx], args.sigma)
        e0 = Get_Overlap(count_vector0, 1-watermark, k_prime0, k_prime0, args.alpha, perturbation_array[idx], args.sigma)
        error = min(k_prime1-e1, k_prime0-e0)
        certified_ba_array[idx] = (config.message_length - 2 * error) / config.message_length

    return certified_ba_array


def regression(save_watermarks, watermark, args, config):
    # Use regression based smoothing
    save_watermarks = (save_watermarks >= 0.5)
    bitacc = np.zeros(args.num_noise)
    for i in range(args.num_noise):
        bitacc[i] = np.sum(watermark == save_watermarks[i], axis=1) / config.message_length

    perturbation_array = np.linspace(0, args.range, int(args.range*100))
    certified_ba_array = np.zeros(int(args.range*100))
    bitacc.sort()
    for idx in range(int(args.range*100)):
        k = certify(perturbation_array[idx], args)
        if k == -1:
            continue
        else:
            certified_ba_array[idx] = bitacc[k]

    return certified_ba_array