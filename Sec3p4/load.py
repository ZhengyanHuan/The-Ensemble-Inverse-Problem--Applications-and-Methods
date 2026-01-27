import numpy as np
import torch
import os
import configs

OUTPUT_DIR_GROUP = [
    "FlatVel-A",
    "FlatVel-B",
    "CurveVel-A",
    "CurveVel-B",
    "FlatFault-A",
    "FlatFault-B",
    "CurveFault-A",
    "CurveFault-B",
    "Style-A",
    "Style-B",
]

TARGET_FILES_type1 = ["data1.npy", "data2.npy", "data3.npy", "data4.npy",
                      "data5.npy", "data6.npy"
                      ]

TARGET_FILES_type2 = ["seis2_1_0.npy", "seis2_1_1.npy", "seis2_1_2.npy", "seis2_1_3.npy",
                      "seis2_1_4.npy", "seis2_1_5.npy"
                      ]

TARGET_FILES_type3 = ["seis6_1_0.npy", "seis6_1_1.npy", "seis6_1_2.npy", "seis6_1_3.npy",
                      "seis6_1_4.npy", "seis6_1_5.npy"
                      ]

TARGET_FILES_GROUP = [
    TARGET_FILES_type1,
    TARGET_FILES_type1,
    TARGET_FILES_type1,
    TARGET_FILES_type1,
    TARGET_FILES_type2,
    TARGET_FILES_type3,
    TARGET_FILES_type2,
    TARGET_FILES_type3,
    TARGET_FILES_type1,
    TARGET_FILES_type1,
]

def load_dataset(type, load_num = len(OUTPUT_DIR_GROUP)):

    # type = ["seis", "vel"]
    # M: number of priors
    # N: number of samples in one prior
    # S: number of sources
    # T: number of time steps in detection
    # D: number of receivers
    xMNSTD = None
    # print(load_num)
    for i in range(load_num):
        xNSTD = None
        for j in range(len(TARGET_FILES_type1)):
            dir = os.path.join(OUTPUT_DIR_GROUP[i], TARGET_FILES_GROUP[i][j])
            if type == "seis":
                data_np = np.load("dataset/data/"+dir)
            elif type == "vel":
                data_np = np.load("dataset/vel/"+dir.replace("seis","vel").replace("data","model"))
            else:
                raise ValueError
            if xNSTD is None:
                xNSTD = data_np.copy()
            else:
                xNSTD = np.concatenate((xNSTD, data_np.copy()), axis=0)

        if xMNSTD is None:
            xMNSTD = xNSTD.copy()[np.newaxis,...]
        else:
            xMNSTD = np.concatenate((xMNSTD, xNSTD.copy()[np.newaxis,...]), axis=0)
        # print(xNSTD.shape)
    return xMNSTD


def normalize(x, mean, std):
    return (x - mean) / std


class SMILE_dataset:
    def __init__(self, p = 0.9):
        yMNSTD_np = load_dataset(type="seis")
        xMN1DD_np = load_dataset(type="vel")

        self.y_std = np.std(yMNSTD_np)
        self.x_std = np.std(xMN1DD_np)
        self.y_mean = np.mean(yMNSTD_np)
        self.x_mean = np.mean(xMN1DD_np)
        self.M = yMNSTD_np.shape[0]
        self.dr_prev = np.max(xMN1DD_np) - np.min(xMN1DD_np)

        yMNSTD_np = normalize(yMNSTD_np, self.y_mean, self.y_std)
        xMN1DD_np = normalize(xMN1DD_np, self.x_mean, self.x_std)

        yMNSTD_tensor = torch.from_numpy(yMNSTD_np).to(configs.device)
        xMN1DD_tensor = torch.from_numpy(xMN1DD_np).to(configs.device)

        thres = int(yMNSTD_tensor.shape[1] * p)

        self.y_train = yMNSTD_tensor[:, :thres]
        self.y_test = yMNSTD_tensor[:, thres:]
        self.x_train = xMN1DD_tensor[:, :thres]
        self.x_test = xMN1DD_tensor[:, thres:]
        self.dr = xMN1DD_tensor.amax().item()-xMN1DD_tensor.amin().item()

    def SMILE_normalize_x(self, x):
        return (x - self.x_mean) / self.x_std

    def SMILE_normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def SMILE_denormalize_x(self, x):
        return (x * self.x_std) + self.x_mean

    def SMILE_denormalize_y(self, y):
        return (y * self.y_std) + self.y_mean

    def get_mixed_data(self, batch_size, exclude = None):
        if exclude is None:
            i = np.random.randint(self.M)
        else:
            available_ind = np.linspace(0,self.M-1,self.M).astype(int)
            available_ind = available_ind[~np.isin(available_ind, exclude)]
            i = np.random.choice(available_ind)

        selected_samples = np.random.choice(self.x_train.shape[1], batch_size, replace=False)

        return self.x_train[i][selected_samples], self.y_train[i][selected_samples]

    def get_test_data(self, batch_size, category):
        i = np.random.choice(category)
        selected_samples = np.random.choice(self.x_test.shape[1], batch_size, replace=False)

        return self.x_test[i][selected_samples], self.y_test[i][selected_samples]


            
