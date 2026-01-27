#%%
import os
import tqdm
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
# from torch.optim.lr_scheduler import LinearLR
from model import Model
from model_v import Model_vanilla
import configs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--dist_info', action='store_true')
    parser.add_argument('--info_dim', default=-1, type=int, required=False)
    parser.add_argument('--save_name', default='', type=str, required=False)

    
    args = parser.parse_args()

    print('model_type:', args.model_type)
    print('dist_info:', args.dist_info)
    if args.info_dim == -1:
        args.info_dim = configs.info_dim
    configs.set_seed(configs.seed)
    print('info dim:', args.info_dim)

    if not os.path.exists(configs.plots_path):
       os.makedirs(configs.plots_path)
    if not os.path.exists(configs.ckpt_path):
       os.makedirs(configs.ckpt_path)

    print(configs.input_path+'    '+configs.state_name+configs.exparameter + '   '+configs.data_type + '    ' + configs.moments_info)

    # load training data
    # x is truth, y is reco (detector)
    if configs.data_type == 'synthetic':
        x = np.load(configs.input_path + "train_truth_" + configs.train_type + '_'+configs.exparameter+ ".npy", mmap_mode='r')
        y = np.load(configs.input_path + "train_reco_" + configs.train_type + '_'+configs.exparameter + ".npy", mmap_mode='r')
    else:
        x = np.load(configs.input_path + "train_truth_combined18_xsmall_organized"  + ".npy", mmap_mode='r')
        y = np.load(configs.input_path + "train_reco_combined18_xsmall_organized" + ".npy", mmap_mode='r')

    if configs.moments == "_no_moments":
        x = x[:,:configs.data_dim]
        y = y[:, :configs.data_dim]
    else:
        if configs.moments_in_x == True:
            pass
        else:
            x = x[:,:configs.data_dim]

    x_train = torch.tensor(x, dtype=torch.float32).to(configs.device)
    y_train = torch.tensor(y, dtype=torch.float32).to(configs.device)
    # use only 90% of data for training
    # x_train = x[:int(0.90*x.shape[0]),:]
    # y_train = y[:int(0.90*y.shape[0]),:]

    # use other 10% of data for validation
    # x_val = x[int(0.90*x.shape[0]):,:]
    # y_val = y[int(0.90*y.shape[0]):,:]


    # initiate model and optimizer
    # cDDPM = torch.compile(Model(configs.device, configs.beta_1, configs.beta_T, configs.T, configs.shape_in[0], configs.shape_out[0]))
    if args.dist_info:
        cDDPM = Model(configs.device, configs.beta_1, configs.beta_T, configs.T, configs.shape_in[0], configs.shape_out[0],configs.batch_size, args.info_dim)
    else:
        cDDPM = Model_vanilla(configs.device, configs.beta_1, configs.beta_T, configs.T, configs.shape_in[0], configs.shape_out[0],configs.batch_size)



    #if continue to train
    # state_dict = torch.load(r'./model-state/FMcombined18_no_moments_real25_batchsize2000_it5000_FM/FM_gen_FMcombined18_no_moments_b2000_it1000_na_t.pth')
    # unwanted_prefix = '_orig_mod.'
    # for k,v in list(state_dict.items()):
    #     if k.startswith(unwanted_prefix):
    #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # cDDPM.load_state_dict(state_dict)

    optim = torch.optim.Adam(cDDPM.parameters(), lr = configs.lr)

    # keep track of loss (for plotting)
    train_loss_list = np.zeros(configs.epochs)
    # val_loss_list = []
    epoch_list = []
    step_list = []
    # number of batches
    batch_size = configs.batch_size
    # batches = int(x_train.shape[0]/batch_size)
    # batches_val = int(x_val.shape[0]/batch_size)

    # scheduler = LinearLR(optim, 1, 1e-2, total_iters=configs.epochs*configs.minibatch_num_periter_perdist*configs.dist_num)


    cum_train_loss = 0
    cum_val_loss = 0
    train_step = 0


    # make a list of batches in device so we don't have to load each time

    # all_batches = []
    # for batch in range(batches):
    #     x_batch = torch.tensor(x_train[batch*batch_size:(batch+1)*batch_size], dtype=torch.float32, device=configs.device)
    #     y_batch = torch.tensor(y_train[batch*batch_size:(batch+1)*batch_size], dtype=torch.float32, device=configs.device)
    #     all_batches.append((x_batch, y_batch))

    # val_batches = []
    # for batch in range(batches_val):
    #     x_batch_val = torch.tensor(x_val[batch*batch_size:(batch+1)*batch_size], dtype=torch.float32, device=configs.device)
    #     y_batch_val = torch.tensor(y_val[batch*batch_size:(batch+1)*batch_size], dtype=torch.float32, device=configs.device)
    #     val_batches.append((x_batch_val, y_batch_val))


    pbar = tqdm.tqdm(total=configs.epochs)

    # start training loop
    for iteration in range(1, configs.epochs+1):
        cum_train_loss = 0

        # define training batches
        for batch in range(configs.minibatch_num_periter_perdist):
            for dist in range(configs.dist_num):
                train_step += 1

                ## train the model
                indices = torch.randperm(x_train.shape[0]//configs.dist_num)[:configs.batch_size]

                x_batch = x_train[dist*configs.totnum_perdist:(dist+1)*configs.totnum_perdist,:][indices]
                y_batch = y_train[dist*configs.totnum_perdist:(dist+1)*configs.totnum_perdist,:][indices]
                # x_batch, y_batch = all_batches[batch]
                # print(y_batch)
                optim.zero_grad()
                loss = cDDPM.loss_fn(x_batch,y_batch, model_type=args.model_type)
                loss.backward()
                optim.step()
                # scheduler.step()


                # add to cumulative loss
                cum_train_loss = cum_train_loss + loss.item()

        train_loss = cum_train_loss/(configs.minibatch_num_periter_perdist*configs.dist_num)
        train_loss_list[iteration-1] = train_loss
        
        if iteration == 5 or iteration % configs.save_int == 0:
            # save model state checkpoint
            if args.dist_info:
                torch.save(cDDPM.state_dict(), configs.ckpt_path + args.model_type +'_gen_' + configs.train_type + '_b' + str(batch_size) + '_it' + str(iteration) + '_'+configs.exparameter+'_'+ configs.infonet_type +args.save_name+'.pth')
            else:
                torch.save(cDDPM.state_dict(),
                           configs.ckpt_path + args.model_type + '_gen_' + configs.train_type + '_v' + str(
                               batch_size) + '_it' + str(
                               iteration) + '_' + configs.exparameter + '_' + configs.infonet_type +args.save_name+ '.pth')

            np.save(configs.ckpt_path + args.model_type + '_gen_' + configs.train_type + '_v' + str(
                               batch_size) + '_it' + str(
                               configs.epochs) + '_' + configs.exparameter + '_' + configs.infonet_type +args.save_name+'_loss.npy', train_loss_list)
            # cum_val_loss = 0
            # cum_test_loss = 0
            # for batch in range(batches_val):
            #
            #     x_batch_val, y_batch_val = val_batches[batch]
            #
            #     with torch.no_grad():
            #         loss_val = cDDPM.loss_fn(x_batch_val, y_batch_val)
            #
            #     # add to cumulative loss
            #     cum_val_loss = cum_val_loss + loss_val.item()



        # val_loss = cum_val_loss/batches_val

        # save loss values and reset cumulative loss count every print_int iterations
        # if iteration % configs.save_int == 0:
        #     epoch_list.append(iteration)
        #     step_list.append(train_step)
        #     train_loss_list.append((train_loss))
            # val_loss_list.append((val_loss))

        pbar.update()
        # pbar.set_description(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}")
        pbar.set_description(f"train loss: {train_loss:.6f}")


    #%%
    # plot loss
    plt.clf()
    plt.plot(epoch_list, train_loss_list)
    # plt.plot(epoch_list, val_loss_list)
    # plt.legend(["train loss", "validation loss"])
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss')
    plt.savefig(configs.plots_path + "loss_" + configs.train_type + "_it" +str(configs.epochs) + ".png")