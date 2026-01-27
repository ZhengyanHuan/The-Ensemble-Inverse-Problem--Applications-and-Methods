import configs
import numpy as np
x = np.load(configs.input_path + "train_truth_combined18_xsmall"  + ".npy", mmap_mode='r')
y = np.load(configs.input_path + "train_reco_combined18_xsmall" + ".npy", mmap_mode='r')


uniq_val = np.unique(x[:,-1])
rearanged_x = np.zeros_like(x)
rearanged_y = np.zeros_like(y)
point = 0
for i in range(len(uniq_val)):
    list_tmp = np.where(x[:,-1] == uniq_val[i])[0]
    len_list_tmp = len(list_tmp)
    rearanged_x[point:point+len_list_tmp] = x[list_tmp]
    rearanged_y[point:point+len_list_tmp] = y[list_tmp]
    point += len_list_tmp
    print(point)

np.save(configs.input_path + "train_truth_combined18_xsmall_organized"  + ".npy", rearanged_x)
np.save(configs.input_path + "train_reco_combined18_xsmall_organized" + ".npy", rearanged_y)

