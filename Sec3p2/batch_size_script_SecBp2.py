import FM_class
from evaluate_nosave import eval_models_b, eval_models_b2
import numpy as np

batch_size_group = [5, 10, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000]

epoches = 80000

# batch_size_group = [5, 10]
#
# epoches = 10

# for i in range(len(batch_size_group)):
#     batch_size = batch_size_group[i]
#     tFM = FM_class.tFM(save_int=epoches, tFM_name="tFMb_"+str(batch_size), batch_size=batch_size )
#     tFM.train(epoches)

# total_len = len(batch_size_group)
# tFM_list = [None]*total_len
#
# for i in range(len(batch_size_group)):
#     batch_size = batch_size_group[i]
#     tFM = FM_class.tFM(save_int=epoches, tFM_name="tFMb_"+str(batch_size), batch_size=batch_size )
#     tFM.load_ckpt('./saved_model/tFMb_'+str(batch_size)+'_it_'+str(epoches)+'.pth', )
#     tFM_list[i] = tFM
#
#
# eval_models_b(tFM_list, 40000, [False]*total_len, batch_size_group, rho_list = np.array([-0.9,-0.5,0,0.5,0.9]))



tFM = FM_class.tFM(save_int=epoches, tFM_name="tFMb_"+str(4000), batch_size=4000 )
tFM.load_ckpt('./saved_model/tFMb_'+str(4000)+'_it_'+str(epoches)+'.pth', )

batch_size_group2 = [5, 10, 50, 100, 200, 500, 1000, 2000, 3000, 3500, 4000]
eval_models_b2(tFM, 40000, 4000, np.array([-0.9,-0.5,0,0.5,0.9]), batch_size_group2)



