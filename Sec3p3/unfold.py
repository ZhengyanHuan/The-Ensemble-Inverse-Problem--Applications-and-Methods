#%%
import torch
import tqdm
import os
import numpy as np
from model import Model
from diffusion import DiffusionProcess
from diffusion import FMProcess
import configs
import argparse
from model_v import Model_vanilla


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--dist_info', action='store_true')
    parser.add_argument('--info_dim', default=-1, type=int, required=False)


    args = parser.parse_args()

    configs.set_seed(configs.seed)
    if args.info_dim == -1:
        args.info_dim = configs.info_dim
    configs.set_seed(configs.seed)
    print('info dim:', args.info_dim)
    # cleanup for loading saved state from compiled model
    # state_dict = torch.load(configs.ckpt_path + 'gen_' + configs.state_name + '.pth', weights_only=True)

    if args.dist_info:
        unfold_state_name = configs.train_type + '_b' + str(configs.batch_size) + '_it' + str(configs.epochs)
    else:
        unfold_state_name = configs.train_type + '_v' + str(configs.batch_size) + '_it' + str(configs.epochs)
    state_dict = torch.load(configs.ckpt_path + args.model_type +'_gen_combined18_no_moments_b2000_it9000_na_t_k'+str(args.info_dim)+'.pth')
    print(configs.ckpt_path + args.model_type +'_gen_combined18_no_moments_b2000_it9000_na_t_k'+str(args.info_dim)+'.pth')
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # initiate models and load saved state
    if args.dist_info:
        cDDPM = Model(configs.device, configs.beta_1, configs.beta_T, configs.T, configs.shape_in[0], configs.shape_out[0], configs.batch_size, args.info_dim)
    else:
        cDDPM = Model_vanilla(configs.device, configs.beta_1, configs.beta_T, configs.T, configs.shape_in[0],
                      configs.shape_out[0], configs.batch_size)

    # print(configs.shape_in[0])
    cDDPM.load_state_dict(state_dict)
    if args.model_type == 'DDPM':
        process = DiffusionProcess(configs.beta_1, configs.beta_T, configs.T, cDDPM, configs.device, configs.shape_out, dist_info_required=args.dist_info)
    elif args.model_type == 'FM':
        process = FMProcess(configs.T, cDDPM, configs.device, configs.shape_out, dist_info_required=args.dist_info)
    else:
        raise ValueError('model_type must be DDPM or FM')

    if not os.path.exists(configs.output_path):
       os.makedirs(configs.output_path)

    # print("model: ", configs.ckpt_path + 'gen_' + configs.state_name +'_' +configs.exparameter+'.pth')

    exp_sacle_list_unfold = configs.exp_sacle_list
    for i in range(len(exp_sacle_list_unfold)):
        exp_sacle_list_unfold[i] = "exp" + str(exp_sacle_list_unfold[i]).replace('.','p')

    if configs.data_type == 'synthetic':
        for exp_unfold in exp_sacle_list_unfold:
            print("test data: ", exp_unfold)

            print("loading data...")

            reco = np.load(configs.input_path + "reco_" + configs.train_type + '_' + exp_unfold + ".npy", mmap_mode='r')

            reco = torch.from_numpy(np.array(reco, copy=True)).float().to(configs.device)
            # print(reco.shape)
            if configs.moments == '_has_moments':
                pass
            else:
                reco = reco[:,:configs.data_dim]

            if configs.data_type == 'real':
                reco = reco[:,1:]



            print("unfolding...")

            unfolded = []
            n = reco.shape[0]//configs.sample_size
    #         pbar = tqdm.tqdm(total=n)

            with torch.no_grad():
                for i in (tqdm.tqdm(range(n))):
                    unfolded_part = process.sampling(configs.sample_size, reco[i*configs.sample_size:(i+1)*configs.sample_size])
                    unfolded.append(unfolded_part)
    #                 pbar.update()

            unfolded = torch.cat(unfolded)
            unfolded = unfolded.cpu().numpy()
            # if configs.data_type == "synthetic":
            unfolded = unfolded[:,:configs.data_dim]
    #         pbar=0

            ## undo normalization
            unfolded = unfolded*configs.norm_vec
            ## save unfolded results

            np.save(configs.output_path + "unfold_" + exp_unfold + ".npy", unfolded)

    else:
        # print('xxxxxxxxxxxxxxxxx')
        # real_dataset_name = 'lepqua_CT14lo_part'
        # real_dataset_name = 'ttbar_CT14lo_vincia_part'
        # real_dataset_name = 'wjets_CT14lo_part'
        # real_dataset_name = 'zjets_NNPDF23lo0130_part'

        for dataset_name in ['lepqua_NNPDF23lo0130', 'ttbar_CT14lo_vincia', 'wjets_CT14lo', 'zjets_CTEQ6L1']:
        # for dataset_name in ['combined18_xsmall_organized']:
            print("test data: ", dataset_name)

            print("loading data...")
            reco = np.load(configs.input_path + "reco_" + dataset_name+ ".npy",
                               mmap_mode='r')/configs.norm_vec
            # print(reco)

            reco = torch.from_numpy(np.array(reco, copy=True)).float().to(configs.device)
            # print(reco)

            if configs.moments == '_has_moments':
                pass
            else:
                if configs.data_type == 'real':
                    reco = reco[:, :configs.data_dim]
                else:
                    reco = reco[:,:configs.data_dim]





            print("unfolding...")

            unfolded = []
            n = reco.shape[0]//configs.sample_size
            pbar = tqdm.tqdm(total=n)

            with torch.no_grad():
                for i in range(n):
                    # print(reco[i*configs.sample_size:(i+1)*configs.sample_size].shape)
                    # print(reco[i * configs.sample_size:(i + 1) * configs.sample_size])
                    unfolded_part = process.sampling(configs.sample_size, reco[i*configs.sample_size:(i+1)*configs.sample_size])
                    unfolded.append(unfolded_part)
                    pbar.update()

            unfolded = torch.cat(unfolded)
            unfolded = unfolded.cpu().numpy()
            # if configs.data_type == "synthetic":
            unfolded = unfolded[:,:configs.data_dim]
            pbar=0

            ## undo normalization
            unfolded = unfolded*configs.norm_vec
            ## save unfolded results
            np.save(configs.output_path + "unfold_" + dataset_name + 'k'+str(args.info_dim)+".npy", unfolded)


    print("done!")

