#%%
import numpy as np
import matplotlib.pyplot as plt
import plot_func
import torch


viz = False
var = 1
# rho_list_train = [-0.7,-0.4,0.4,0.7]
# rho_list_test = [-0.9,-0.2,0,0.2,0.9]
rho_list_test = [-0.9]
mix_rho_list_test = [-0.9, 0.9]
A = np.array([[2, 1], [1, 4]])*0.5
mu = np.array([0, 0])  # Mean vector
# cov = var ** 2 * np.array([[1, rho], [rho, 1]])  # Covariance matrix
num_samples_N = int(2e5)
path = './data/'
# samples = np.random.multivariate_normal(mu, cov, num_samples)


def rho_func():
    interval_choice = np.random.rand() < 0.5
    if interval_choice:
        return np.random.uniform(-0.75, -0.25)
    else:
        return np.random.uniform(0.25, 0.75)


def generate_data(batch_size, rho = None):
    if rho is None:
        rho = rho_func()
    cov = var ** 2 * np.array([[1, rho], [rho, 1]])
    samples_original = np.random.multivariate_normal(mu, cov, batch_size)
    samples_perturbed = perturb(samples_original, A, noise_func)

    return samples_original, samples_perturbed, rho

#%%


def noise_func(samples, std_multipler = 0.5, amplitude = 0.2):
    return samples * (1+amplitude) + np.random.randn(*samples.shape) * np.linalg.norm(samples, axis=1)[:,None] * std_multipler


def perturb(samples, A, noise_func):
    out = samples@A
    out += noise_func(samples)
    return out

if __name__ == '__main__':
    name = 'combined_rho'
    samples_original = np.zeros((num_samples_N, 2))
    samples_perturbed = np.zeros((num_samples_N, 2))
    for i in range(num_samples_N):
        # rho_c = rho_func()
        # cov_combined = np.array([[1, rho_c], [rho_c, 1]])
        samples_original[i], samples_perturbed[i], rho = generate_data(1)

    np.save(path+name+'_original_N2.npy', samples_original)
    np.save(path+name+'_perturbed_N2.npy', samples_perturbed)

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


    # M = len(rho_list_test)
    # for i in range(len(rho_list_test)):
    #     rho = rho_list_test[i]
    #     cov = var ** 2 * np.array([[1, rho], [rho, 1]])
    #     samples_original = np.random.multivariate_normal(mu, cov, num_samples_N)
    #     samples_perturbed = perturb(samples_original, A, noise_func)
    #
    #
    #     if viz:
    #         plot_func.plot_scatter_with_info(samples_original[:1000])
    #         plot_func.plot_scatter_with_info(samples_perturbed[:1000])
    #
    #     name = str(rho).replace('.', 'p')
    #
    #
    #     np.save(path+name+'_original_N2.npy', samples_original)
    #     np.save(path+name+'_perturbed_N2.npy', samples_perturbed)

    # num_samples_per_cat = num_samples_N // len(rho_list_train)
    # mix_samples_original = np.zeros([num_samples_N,2])
    # mix_samples_perturbed = np.zeros([num_samples_N,2])
    # for i in range(len(mix_rho_list_test)):
    #     rho = mix_rho_list_test[i]
    #     cov = var ** 2 * np.array([[1, rho], [rho, 1]])
    #     samples_original = np.random.multivariate_normal(mu, cov, num_samples_per_cat)
    #     samples_perturbed = perturb(samples_original, A, noise_func)
    #
    #     mix_samples_original[i*num_samples_per_cat:(i+1)*num_samples_per_cat] = samples_original
    #     mix_samples_perturbed[i*num_samples_per_cat:(i+1)*num_samples_per_cat] = samples_perturbed
    #
    # random_indices = np.random.permutation(np.arange(num_samples_N))
    # mix_samples_original = mix_samples_original[random_indices]
    # mix_samples_perturbed = mix_samples_perturbed[random_indices]
    # name = ''
    # for rho in mix_rho_list_test:
    #     name += str(rho).replace('.','p')
    #
    # np.save(path+name+'_original_N2.npy', mix_samples_original)
    # np.save(path+name+'_perturbed_N2.npy', mix_samples_perturbed)
    #
    # if viz:
    #     plot_func.plot_scatter_with_info(mix_samples_original[:3000])
    #     plot_func.plot_scatter_with_info(mix_samples_perturbed[:3000])