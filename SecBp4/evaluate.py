import matplotlib.pyplot as plt
import numpy as np
import torch
import plot_func
import configs
import tqdm
import os


def unfold_evaluate_plot(model, model_name, truth_N2_name, perturbed_N2_name, path = configs.data_path, batch_size = configs.batch_size, output_path = configs.output_path, plot_num = 10000):
    truth_N2 = torch.tensor(np.load(os.path.join(path, truth_N2_name)), dtype=torch.float32, device=configs.device)
    perturbed_N2 = torch.tensor(np.load(os.path.join(path, perturbed_N2_name)), dtype=torch.float32, device=configs.device)

    n = perturbed_N2.shape[0]//batch_size
    unfolded = []
    with torch.no_grad():
        for i in (tqdm.tqdm(range(n))):
            unfolded_part = model.sampling(batch_size, perturbed_N2[i*batch_size:(i+1)*batch_size])
            unfolded.append(unfolded_part)

        unfolded = torch.cat(unfolded)
        unfolded = unfolded.cpu().numpy()

    np.save(output_path + model_name+"_unfold_" + str(truth_N2_name) + ".npy", unfolded)
    # plt.subplot(1,3,1)
    plot_func.plot_scatter_with_info(unfolded[:plot_num], truth_N2[:plot_num].cpu().numpy() , info='recovered')
    # plt.subplot(1,3,2)
    plot_func.plot_scatter_with_info(perturbed_N2[:plot_num].cpu().numpy(), truth_N2[:plot_num].cpu().numpy() , info = 'perturbed' )
    # plt.subplot(1,3,3)
    plot_func.plot_scatter_with_info(truth_N2[:plot_num].cpu().numpy(), truth_N2[:plot_num].cpu().numpy() , info = 'truth' )
    # plt.show()
    return unfolded


def unfold(model, model_name, truth_N2_name, perturbed_N2_name, path = configs.data_path, batch_size = configs.batch_size, output_path = configs.output_path):

    perturbed_N2 = torch.tensor(np.load(os.path.join(path, perturbed_N2_name)), dtype=torch.float32, device=configs.device)

    n = perturbed_N2.shape[0]//batch_size
    unfolded = []
    with torch.no_grad():
        for i in (tqdm.tqdm(range(n))):
            unfolded_part = model.sampling(batch_size, perturbed_N2[i*batch_size:(i+1)*batch_size])
            unfolded.append(unfolded_part)

        unfolded = torch.cat(unfolded)
        unfolded = unfolded.cpu().numpy()

    np.save(output_path + model_name + "_unfold_" + str(truth_N2_name) , unfolded)
    return unfolded


def evaluate_MSE(truth_N2_name, perturbed_N2_name, unfolded_N2_name,datapath = configs.data_path,  output_path = configs.output_path, tAndr = False):
    truth_N2 = np.load(os.path.join(datapath, truth_N2_name))
    # print(truth_N2.shape)
    perturbed_N2 = np.load(os.path.join(datapath, perturbed_N2_name))
    unfolded_N2 = np.load(os.path.join(output_path, unfolded_N2_name))

    MSE = np.mean(np.linalg.norm(truth_N2 - unfolded_N2, axis = 1) ** 2)
    print('MSE = between truth and recovered', str(MSE.item()))

    if tAndr:
        MSE = np.mean(np.linalg.norm(truth_N2 - perturbed_N2, axis=1) ** 2)
        print('MSE between truth and perturbed is= ', str(MSE.item()))


def evaluateSWD(truth_N2_name, perturbed_N2_name, unfolded_N2_name,datapath = configs.data_path,  output_path = configs.output_path, plot = False, plot_truth = False, plot_num = 10000, rep_num = 50, tAndr = False):
    truth_N2 = np.load(os.path.join(datapath, truth_N2_name))
    # print(truth_N2.shape)
    perturbed_N2 = np.load(os.path.join(datapath, perturbed_N2_name))
    unfolded_N2 = np.load(os.path.join(output_path, unfolded_N2_name))


    uSWD_list = np.zeros(rep_num)

    for j in range(rep_num):
        select_ind_truth = np.random.choice(truth_N2.shape[0], plot_num, replace=False)
        # select_ind_unfold = np.random.choice(unfolded_N2.shape[0], plot_num, replace=False)
        uSWD = plot_func.compute_SWD(truth_N2[select_ind_truth], unfolded_N2[select_ind_truth], sample_size=None)
        uSWD_list[j] = uSWD

    print("SWD between truth and recovered: mean " + str(np.mean(uSWD_list)) +", std " + str(np.std(uSWD_list)) + ' rep_num: ' + str(rep_num))

    if tAndr:
        pSWD = plot_func.compute_SWD(truth_N2[:plot_num], perturbed_N2[:plot_num], sample_size=None)
        print("SWD between truth and perturbed is: "+ str(pSWD.item()))

    
    if plot:
        plot_func.plot_scatter_with_info(unfolded_N2[:plot_num], truth_N2[:plot_num] , info='', distance=False, save=True, save_name='recovered')
    if plot_truth:
        plot_func.plot_scatter_with_info(truth_N2[:plot_num], info='', distance=False, save=True, save_name='Truth')

    # plt.subplot(1,3,2)
    #     plot_func.plot_scatter_with_info(perturbed_N2[:plot_num], truth_N2[:plot_num] , info = 'perturbed' )
        # plt.subplot(1,3,3)
        # plot_func.plot_scatter_with_info(truth_N2[:plot_num], truth_N2[:plot_num] , info = 'truth' )
    
    # return pSWD, uSWD
    
    
    