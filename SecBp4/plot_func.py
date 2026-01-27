import matplotlib.pyplot as plt
import os
from scipy.stats import wasserstein_distance
from ot.sliced import sliced_wasserstein_distance as SWD

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from scipy.stats import gaussian_kde
import configs
import numpy as np
import ot




def plot_scatter_with_info(xy, refer_samples = [] , info = '',save_name = '', distance=True, show =True, save = False):
    #     plt.figure(figsize=(6, 6))
    z = gaussian_kde(xy.transpose())(xy.transpose())
    # z = uniform_kde(xy, xy, 0.1)
    # heatmap, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=30, range=[[0, 5], [0, 5]])
    # Create the plot

    fig, ax = plt.subplots()
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=z, s=3, cmap='viridis')

    # ax.add_patch(square)
    plt.colorbar(scatter, label='Density')
    ax.set_aspect('equal')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-6,6)
    plt.ylim(-6,6)

    if distance == True and refer_samples.__len__()!=0:
        distance = compute_SWD(refer_samples, xy)
        title =  r'  SWD:%.4f' % distance
        title = info + title
    else:
        title =  ' '
        title = info + title

    plt.title(title)
    plt.grid(True)
    if save:
        plt.savefig('./fig/' + save_name + '.png', bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()


def only_info(xy, uniform_samples, distance=True):
    pass
    # num_out, prob = prob_out2(xy)
    #
    # if distance == True:
    #     distance = compute_SWD(uniform_samples, xy.transpose())

    # return num_out, prob, distance




def compute_dist(uniform_samples, generated_samples, numItermax=100000):
    n_samples = int(uniform_samples.shape[0])
    M = ot.dist(uniform_samples, generated_samples)

    uniform_weights = np.ones(n_samples) / n_samples
    point_weights = np.ones(n_samples) / n_samples

    # Compute the 2D Wasserstein distance (optimal transport cost)
    emd_2d = ot.emd2(uniform_weights, point_weights, M, numItermax=numItermax)
    return emd_2d


def compute_SWD(ref, pred, sample_size=None):
    sample_size = min(pred.shape[0], ref.shape[0]) if sample_size is None else sample_size
    pred = shuffle(pred, sample_size=sample_size)
    ref = shuffle(ref, sample_size=sample_size)

    return SWD(pred, ref)


def shuffle(x, sample_size):
    """
        x: (B, D)
        ===
        return: (sample_size, D)
    """
    idx = np.random.choice(x.shape[0], sample_size, replace=False)
    return x[idx]


