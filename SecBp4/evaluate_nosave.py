import numpy as np
import torch
import plot_func
import configs
import tqdm
from dataset import generate_data
from plot_func import plot_scatter_with_info

def unfold_batch(model, y):
    batch_size = y.shape[0]
    with torch.no_grad():
        unfolded_part = model.sampling(batch_size, y)

    return unfolded_part


def eval_models(model_list, batch_num, require_rho_list, batch_size = None, rho_list = None, mu1_list = None, mu2_list = None):
    if rho_list is None:
        rho_list = np.linspace(-1,1,21)
    if mu1_list is None:
        mu1_list = np.linspace(-2.5,2.5,41)
    if mu2_list is None:
        mu2_list = np.linspace(-2.5,2.5,41)

    if batch_size is None:
        batch_size = configs.batch_size

    MSE_record = np.zeros([len(model_list), len(rho_list), len(mu1_list), len(mu2_list)])
    SWD_record = np.zeros([len(model_list), len(rho_list), len(mu1_list), len(mu2_list)])

    for rho_index in tqdm.tqdm(range(len(rho_list))):
        for mu1_index in (range(len(mu1_list))):
            for mu2_index in (range(len(mu2_list))):
                rho = rho_list[rho_index]
                mu1 = mu1_list[mu1_index]
                mu2 = mu2_list[mu2_index]
                unfolded = [[] for _ in range(len(model_list))]
                x_record = []
                for i in range(batch_num):
                    x_batch_B2, y_batch_B2, z = generate_data(batch_size, z = (rho, mu1, mu2))
                    x_batch_B2 = torch.tensor(x_batch_B2).to(configs.device).to(torch.float32)
                    y_batch_B2 = torch.tensor(y_batch_B2).to(configs.device).to(torch.float32)

                    for model_index in range(len(model_list)):
                        if require_rho_list[model_index]:
                            z_tensor = torch.tensor(z, dtype=torch.float32, device=configs.device).repeat(y_batch_B2.shape[0], 1)
                            y_batch_B2_cat = torch.cat([y_batch_B2, z_tensor], dim=1)
                        else:
                            y_batch_B2_cat = y_batch_B2

                        model = model_list[model_index]
                        unfolded_part = unfold_batch(model, y_batch_B2_cat)
                        unfolded[model_index].append(unfolded_part)

                    x_record.append(x_batch_B2)

                x_record = torch.cat(x_record)

                for model_index in range(len(model_list)):
                    # print(unfolded[model_index].__len__())
                    unfolded_compute = torch.cat(unfolded[model_index])
                    MSE_record[model_index, rho_index, mu1_index, mu2_index] = np.mean(np.linalg.norm(x_record.cpu().numpy() - unfolded_compute.cpu().numpy(), axis=1) ** 2)
                    SWD_record[model_index, rho_index, mu1_index, mu2_index] = plot_func.compute_SWD(x_record.cpu().numpy(), unfolded_compute.cpu().numpy(), sample_size= None)

        np.savez('./output/test_record.npz', MSE_record=MSE_record, SWD_record=SWD_record)




def plot_models(model_list, batch_num, require_rho_list, batch_size = None, rho_list = None, save = False):
    if rho_list is None:
        rho_list = [0.9]

    if batch_size is None:
        batch_size = configs.batch_size

    MSE_record = np.zeros([len(model_list), len(rho_list)])
    SWD_record = np.zeros([len(model_list), len(rho_list)])

    for rho_index in tqdm.tqdm(range(len(rho_list))):
        rho = rho_list[rho_index]
        unfolded = [[] for _ in range(len(model_list))]
        x_record = []
        for i in range(batch_num):
            x_batch_B2, y_batch_B2, rho = generate_data(batch_size, rho = rho)
            x_batch_B2 = torch.tensor(x_batch_B2).to(configs.device).to(torch.float32)
            y_batch_B2 = torch.tensor(y_batch_B2).to(configs.device).to(torch.float32)

            for model_index in range(len(model_list)):
                if require_rho_list[model_index]:
                    rho_tensor = torch.full((y_batch_B2.shape[0], 1), rho, dtype=torch.float32, device=configs.device)
                    y_batch_B2_cat = torch.cat([y_batch_B2, rho_tensor], dim=1)
                else:
                    y_batch_B2_cat = y_batch_B2

                model = model_list[model_index]
                unfolded_part = unfold_batch(model, y_batch_B2_cat)
                unfolded[model_index].append(unfolded_part)

            x_record.append(x_batch_B2)

        x_record = torch.cat(x_record)

        plot_scatter_with_info(x_record.cpu().numpy(), info="truth", save_name='truth', save=save)
        for model_index in range(len(model_list)):
            # print(unfolded[model_index].__len__())
            unfolded_compute = torch.cat(unfolded[model_index])
            plot_scatter_with_info(unfolded_compute.cpu().numpy(), info=str(model_index), save=save, save_name=str(model_index))



    #         MSE_record[model_index, rho_index] = np.mean(np.linalg.norm(x_record.cpu().numpy() - unfolded_compute.cpu().numpy(), axis=1) ** 2)
    #         SWD_record[model_index, rho_index] = plot_func.compute_SWD(x_record.cpu().numpy(), unfolded_compute.cpu().numpy(), sample_size= None)
    #
    # np.savez('./output/test_record.npz', MSE_record=MSE_record, SWD_record=SWD_record)

