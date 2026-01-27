#%%
import numpy as np
import matplotlib.pyplot as plt
import plot_func
import torch


viz = False
var = 1

rho_list_test = [-0.9]

A = np.array([[2, 1], [1, 4]])*0.5

num_samples_N = int(1e6)
path = './data/'
# samples = np.random.multivariate_normal(mu, cov, num_samples)


def rho_func():
    interval_choice = np.random.rand() < 0.5
    if interval_choice:
        return np.random.uniform(-0.75, -0.25)
    else:
        return np.random.uniform(0.25, 0.75)

def mu1_func():
    interval_choice = np.random.rand() < 0.5
    if interval_choice:
        return np.random.uniform(-1.5, -0.5)
    else:
        return np.random.uniform(0.5, 1.5)

def mu2_func():
    interval_choice = np.random.rand() < 0.5
    if interval_choice:
        return np.random.uniform(-1.5, -0.5)
    else:
        return np.random.uniform(0.5, 1.5)

def generate_data(batch_size, z =(None, None, None)):
    if z[0] is None:
        rho = rho_func()
    else:
        rho = z[0]
    if z[1] is None:
        mu1 = mu1_func()
    else:
        mu1 = z[1]
    if z[2] is None:
        mu2 = mu2_func()
    else:
        mu2 = z[2]

    mu = np.array([mu1, mu2])
    cov = var ** 2 * np.array([[1, rho], [rho, 1]])
    samples_original = np.random.multivariate_normal(mu, cov, batch_size)
    samples_perturbed = perturb(samples_original, A, noise_func)

    return samples_original, samples_perturbed, (rho, mu1, mu2)

#%%


def noise_func(samples, std_multipler = 0.5, amplitude = 0.2):
    return samples * (1+amplitude) + np.random.randn(*samples.shape) * np.linalg.norm(samples, axis=1)[:,None] * std_multipler


def perturb(samples, A, noise_func):
    out = samples@A
    out += noise_func(samples)
    return out

if __name__ == '__main__':
    # M = len(rho_list_train)
    # dataset_original_MN2 = np.zeros([M, num_samples_N, 2])
    # dataset_perturbed_MN2 = np.zeros([M, num_samples_N, 2])
    # for i in range(len(rho_list_train)):
    #     rho = rho_list_train[i]
    #     cov = var ** 2 * np.array([[1, rho], [rho, 1]])
    #     samples_original = np.random.multivariate_normal(mu, cov, num_samples_N)
    #     samples_perturbed = perturb(samples_original, A, noise_func)
    #
    #
    #     dataset_original_MN2[i, :, :] = samples_original
    #     dataset_perturbed_MN2[i, :, :] = samples_perturbed
    #     if viz:
    #         plot_func.plot_scatter_with_info(samples_original[:1000])
    #         plot_func.plot_scatter_with_info(samples_perturbed[:1000])
    #
    # np.save(path+'combined_original_MN2.npy', dataset_original_MN2)
    # np.save(path+'combined_perturbed_MN2.npy', dataset_perturbed_MN2)
    #
    # with open(path+"combined_setting.txt", "w") as f:
    #     f.write(str(rho_list_train))
    # f.close()


    M = len(rho_list_test)
    for i in range(len(rho_list_test)):
        rho = rho_list_test[i]
        cov = var ** 2 * np.array([[1, rho], [rho, 1]])
        samples_original = np.random.multivariate_normal(mu, cov, num_samples_N)
        samples_perturbed = perturb(samples_original, A, noise_func)


        if viz:
            plot_func.plot_scatter_with_info(samples_original[:1000])
            plot_func.plot_scatter_with_info(samples_perturbed[:1000])

        name = str(rho).replace('.', 'p')


        np.save(path+name+'_original_N2.npy', samples_original)
        np.save(path+name+'_perturbed_N2.npy', samples_perturbed)

