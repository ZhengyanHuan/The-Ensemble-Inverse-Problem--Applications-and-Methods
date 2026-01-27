import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


lr = 3e-4
beta_1 = 1e-4 # 1e-4
beta_T = 0.02 # 0.02
T = 100 # 100
dim_data = 2
dim_info = 3
# dim_out = 2

batch_size = 4000
default_save_int = 1000

ckpt_path = './saved_model/'
data_path = './data/'
output_path = './output/'
cddpm_name = 'cddpm'
tddpm_name = 'tddpm'
cFM_name = 'cFM'
tFM_name = 'tFM'