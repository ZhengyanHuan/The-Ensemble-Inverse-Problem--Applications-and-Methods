import configs
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
import torch.nn as nn

import torch
import matplotlib.pyplot as plt

# import ot
from tqdm import tqdm
# from MNIST_classifier_pretrained import MNIST_classifier
from ot.sliced import sliced_wasserstein_distance as SWD

# lenet_pretrained = MNIST_classifier()


def compute_SWD(ref, pred, sample_size=None):
    sample_size = min(pred.shape[0], ref.shape[0]) if sample_size is None else sample_size
    pred = shuffle(pred, sample_size=sample_size)
    ref = shuffle(ref, sample_size=sample_size)

    return SWD(pred, ref)

def shuffle(x, sample_size):
    """
        x: (B, D)
        ===
        return: (sample_size, D)
    """
    idx = np.random.choice(x.shape[0], sample_size, replace=False)
    return x[idx]



def MNIST_eval_batch(model_list, test_dataset, save_name,model_type_list, clamp = True, eval_num=101, rep_num=10 ):
    ssim_metric = SSIM(data_range=1.0).to(configs.device)
    MSE_pixel_wise_record = np.zeros([len(model_list), eval_num])
    ssim_record = np.zeros([len(model_list), eval_num])

    # correct_rate_record = np.zeros(10)
    alpha_list = np.linspace(0.0, 1.0, num=eval_num)
    for i in tqdm(range(eval_num)):
        alpha = alpha_list[i]
        MSE_pixel_wise = np.zeros(len(model_list))
        ssim = np.zeros(len(model_list))
        SWD = np.zeros(len(model_list))

        for j in range(rep_num):
            test_truth, test_blurred  = test_dataset.get_mixed_data(configs.batch_size, alpha=alpha, train = False)
            test_truth = test_truth.to(configs.device)
            test_blurred = test_blurred.to(configs.device)
            # batch_num = 0
        # for batch1, batch2 in zip(data_loader_blurred_test, data_loader_truth_test):
        #     if len(batch1[0])<configs.batch_size:
        #         continue
        #     else:
        #         batch_num += 1
            for model_index in range(len(model_list)):
                model = model_list[model_index]
                model_type = model_type_list[model_index]

                if model_type == "DDPM":
                    test_recover = model.sampler(model.denoise_model, test_blurred)
                elif model_type == "FM":
                    test_recover = model.sampler_FM(model.denoise_model, test_blurred)
                elif model_type == "SB":
                    test_recover = model.sampler(model.opt, test_blurred).to('cuda:0')
                else:
                    raise NotImplementedError
                if clamp:
                    test_recover = torch.clamp(test_recover, 0.0, 1.0)

                MSE_pixel_wise[model_index] += torch.norm(test_recover -test_truth)**2/(configs.batch_size*28*28)
                ssim[model_index] += ssim_metric(test_recover ,test_truth)


        # print()
        MSE_pixel_wise_record[:,i] = (MSE_pixel_wise/(rep_num))
        ssim_record[:,i] = (ssim/(rep_num))

        # correct_rate_record[i] = (correct_num/(batch_num*configs.batch_size)).cpu().item()
    np.savez(
                    './saved_model/' + save_name + '_' + 'test_record.npz',
                    MSE_pixel_wise_record = MSE_pixel_wise_record,ssim_record = ssim_record)

