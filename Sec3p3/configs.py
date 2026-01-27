import torch
import numpy as np

seed = 24

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)



## CHOOSE TRAINING DATASET AND OPTIONS ##
## ----------------------------------- ##

## generalized training dataset
train_type = 'combined18'
# train_type = 'combined6'
exparameter = 'na' #exp0p2, uniform [0p2_0p1_0p05] [0p7_0p3_0p07]
## option to run with fake or no moments
# moments = "" # default, with moments
moments = "_no_moments"
# moments = "_fake_moments"



## CHOOSE TEST DATASET ##
## ------------------- ##

## test datasets
# exp_scale = 0.7
# exp_scale = 0.5
# exp_scale = 0.3
exp_scale = 0.5
# exp_scale = 0.1
# exp_scale = 0.07
# exp_scale = 0.06
exp_sacle_list = [0.06,0.07,0.2,0.3,0.5,0.7]


unf_type = "exp" + str(exp_scale).replace('.','p')
process_name = 'test'

moments_in_x = False


## --------------------------------- ##
## DO NOT CHANGE ANYTHING BELOW HERE ##
## --------------------------------- ##

data_type = 'real' # synthetic real

train_type = train_type + moments
exp_print = "exp" + str(exp_scale).replace('.','p')

# num of epochs
epochs = 10000
# batch size
batch_size = 2000
# how many jets to unfold
unfold_size = 1000000
# sample size during inference
sample_size = batch_size

# compute validation loss, print loss, and save chekpoint every save_int epochs
save_int = 1000
save_ckpts = True
state_name = train_type + '_b' + str(batch_size) + '_it' + str(epochs)




if moments == "_has_moments":
    if moments_in_x == True:
        moments_info = '_moments_in_x'
    else:
        moments_info = '_no_moments_in_x'
else:
    moments_info = '_tFM'

if data_type == 'synthetic':
    output_path = "./outputs/" + moments_info +exparameter+ "/"
    plots_path = "./plots/" + moments_info +'_' +exparameter+"/"
else:
    output_path = "./outputs/" + '_real25_batchsize' + moments_info + "/"
    plots_path = "./plots/" + 'real25' + moments_info+ "/"
    # plots_path = "./plots/" + 'real_condon_y1/'



if data_type == 'synthetic':
    ckpt_path = 'model-state/' + train_type + '_b' + str(batch_size) + '_it5000' + moments_info +"/"
    input_path = "./datasets/" + moments + "/"
elif data_type == 'real':
    ckpt_path = 'model-state/' + train_type + '_real25_batchsize' + str(batch_size) + '_it5000' + moments_info +"/"
    input_path = "./datasets/" + "ml-unfold-datasets-test" + "/"

## Hyperparameter of diffusion model and perturbation process
lr = 5e-5
beta_1 = 1e-4 # 1e-4
beta_T = 0.02 # 0.02
T = 500 # 100

if data_type == 'synthetic':
    data_dim = 4
else:
    data_dim = 7



if moments == "_no_moments":
    n_dims = data_dim
    shape_in = (n_dims*2,)
    shape_out = (n_dims,)
else:
    n_dims = data_dim + 6  # 4-momentum + 6 moments
    if moments_in_x == True:
        shape_in = (n_dims * 2,)
        shape_out = (n_dims,)
    else:
        shape_in = (n_dims + data_dim,)
        shape_out = (data_dim,)

device = torch.device('cuda:0')

## ranges for normalization
eta_range = 4.4
phi_range = 3.5
pT_range = 1000
E_range = 4000
if data_type == 'synthetic':
    norm_vec = np.array([pT_range, eta_range, phi_range, E_range])
else:
    norm_vec =np.array([pT_range, eta_range, phi_range, E_range, pT_range, pT_range, E_range])
    # norm_vec =np.array([pT_range, eta_range, phi_range, E_range, 1, 1, 1])
#     norm_vec = np.array([1,1,1,1,1,1,1])

# real dataset name
# real_dataset_name = 'lepqua_CT14lo_part'
# real_dataset_name = 'ttbar_CT14lo_vincia_part'
# real_dataset_name = 'wjets_CT14lo_part'
real_dataset_name = 'zjets_NNPDF23lo0130_part'

info_dim = 6
totnum_perdist = int(1e5)
minibatch_num_periter_perdist = 50
if data_type == 'synthetic':
    dist_num = 4
else:
    dist_num = 18

infonet_type = 't' #t: set transformer #s:deep set



