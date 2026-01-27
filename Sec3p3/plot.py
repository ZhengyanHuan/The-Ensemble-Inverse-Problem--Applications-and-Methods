#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.stats import energy_distance
import configs
import pickle
import argparse

plt.rcParams.update({
    'font.size': 13,  # Default font size
    'axes.labelsize': 16,  # Font size for x and y labels
    'axes.titlesize': 18,  # Font size for the title
    'xtick.labelsize': 14,  # Font size for x tick labels
    'ytick.labelsize': 14,  # Font size for y tick labels
    'legend.fontsize': 12  # Font size for the legend
})

def calculate_errors(hist, bin_edges):
    # Calculate statistical errors (sqrt(N) for each bin)
    errors = np.sqrt(hist)
    # Convert to relative errors
    relative_errors = errors / hist
    return relative_errors

k=-1
remove_outliers = True


distance_record = dict()
mse_record = dict()
exp_sacle_list_unfold = configs.exp_sacle_list
for i in range(len(exp_sacle_list_unfold)):
    exp_sacle_list_unfold[i] = "exp" + str(exp_sacle_list_unfold[i]).replace('.','p') 
    
if configs.data_type == 'synthetic':
    for exp_unfold in exp_sacle_list_unfold:
        print("train data type: ", configs.train_type)
        print("test data type: ", exp_unfold)


        ## import data from output files



        # reco = np.load(configs.input_path + "reco_" + configs.exp_print + ".npy")
        # truth = np.load(configs.input_path + "truth_" + configs.exp_print + ".npy")

        reco = np.load(configs.input_path + "test_reco_" + configs.train_type + '_' + exp_unfold + ".npy", mmap_mode='r')
        truth = np.load(configs.input_path + "test_truth_" + configs.train_type + '_' + exp_unfold + ".npy", mmap_mode='r')
#         unfold = np.load("./outputs/" + configs.train_type  + configs.exparameter+ "/unfold_" + exp_unfold + ".npy")
        unfold = np.load("./outputs/" + configs.moments_info  + configs.exparameter+ "/unfold_" + exp_unfold + ".npy")
        print("./outputs/" + configs.moments_info  + configs.exparameter+ "/unfold_" + exp_unfold + ".npy")

        unfold = unfold[:,:configs.data_dim]
        truth = truth[:,:configs.data_dim]
        reco = reco[:,:configs.data_dim]


        n_unf = unfold.shape[0]
        print(n_unf)

        # if not unfolding full data distribution
        reco = reco[:n_unf,:]
        truth = truth[:n_unf,:]

        ## undo normalization
        reco = reco*configs.norm_vec
        truth = truth*configs.norm_vec

        x_text_1 = 0.10
        x_text_2 = 0.40
        y_text_1 = 0.92

        # ranges for plotting
        n_bins = 51

        plot_pT_min = 30
        plot_pT_max = 800
        if configs.exp_scale < 0.1:
            plot_pT_max = 600
        plot_E_min = 0
        plot_E_max = 2000
        plot_eta_min = -4.4
        plot_eta_max = 4.4
        plot_phi_min = -3.2
        plot_phi_max = 3.2

        x_axis_min = [plot_pT_min, plot_eta_min, plot_phi_min, plot_E_min]
        x_axis_max = [plot_pT_max, plot_eta_max, plot_phi_max, plot_E_max]
        y_axis_space = [50, 10, 2, 10]

        ## make plots!!!

        vec = ["pT", "eta", "phi", "E"]
        axis_label = [r'$p_T$ [GeV]', r'$\eta$', r'$\phi$', r'$E$ [GeV]']
        for i in range(len(vec)):
            if i == 0: # pT
                bins = np.array([0, 60, 80, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000])
            if i == 1: # eta
                bins = np.linspace(plot_eta_min, plot_eta_max, n_bins)
            if i == 2: # phi
                bins = np.linspace(plot_phi_min, plot_phi_max, n_bins)
            if i == 3: # E
                bins = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1850, 2000, 2250, 2500, 3000, 4000])

            bin_centers = 0.5*(bins[1:] + bins[:-1])

            if remove_outliers:
                ## get rid of huge outliers
                mask = np.abs(unfold[:,i]) < 100000
                wass_unfold = wasserstein_distance(truth[:,i][mask], unfold[:,i][mask])
                wass_reco = wasserstein_distance(truth[:,i][mask], reco[:,i][mask])
            else:
                wass_unfold = wasserstein_distance(truth[:,i], unfold[:,i])
                wass_reco = wasserstein_distance(truth[:,i], reco[:,i])
            
            distance_record[vec[i]+exp_unfold] = wass_unfold
            print(truth[:,i])
            print(reco[:,i])
            print(unfold[:,i])
            
            plt.clf()

            f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(6, 6))


            # Calculate histograms and errors
            t, _ = np.histogram(truth[:,i], bins=bins, density=False)
            p, _ = np.histogram(unfold[:,i], bins=bins, density=False)
            r, _ = np.histogram(reco[:,i], bins=bins, density=False)

            t_errors = calculate_errors(t, bins)

            # plot truth, reco, and unfold
            a0.hist(truth[:,i], bins=bins, density=True, histtype='step', label='truth',linewidth=3.5, zorder=1)
            a0.hist(unfold[:,i], bins=bins, density=True, histtype='step', label='unfold', linewidth=3, linestyle="dashed", zorder=2)
            a0.hist(reco[:,i], bins=bins, density=True, histtype='step', label='detector', linewidth=3, linestyle="dashed", zorder=3)


            a0.legend(loc='upper right')

            a0.set_xticklabels([])
            a0.set_xlim(x_axis_min[i], x_axis_max[i])
            a0.set_yscale('log')
            a0.set_ylabel(r'$d\sigma/dx$')
            # set extra space for text
            y_min, y_max = a0.get_ylim()
            a0.set_ylim(y_min, y_max*y_axis_space[i])
            if vec[i] == "phi":
              a0.set_ylim(0.05, y_max*y_axis_space[i])

            # add text with metrics 
            a0.text(x_text_1, y_text_1+0.02, r'Wasserstein:', transform=a0.transAxes) 
            a0.text(x_text_1, y_text_1-0.05, r'  unfolded = %.2f' % wass_unfold, transform=a0.transAxes)
            a0.text(x_text_1, y_text_1-0.12, r'  detector = %.2f' % wass_reco, transform=a0.transAxes)
            # print the process name of the unfolded distribution
            a0.text(0.10, 0.06, r"$\beta =$ " + str(exp_unfold), transform=a0.transAxes)





            ## make array of ones like bin_centers
            ones = []
            for j in range(len(bin_centers)):
                ones.append(1)

            ratio_unfold = p / t
            ratio_reco = r / t

            ratio_unfold_errors = ratio_unfold * t_errors
            # ratio_unfold_errors = ratio_unfold * np.sqrt((t_errors)**2 + (calculate_errors(p, bins))**2)


            a1.hist(bin_centers, bins=bins, weights=None, histtype="step", label='',linewidth=2.0, alpha=0.8)
            a1.hist(bin_centers, bins=bins, weights=ratio_unfold, histtype="step", label='unfold/truth',linewidth=3.0)
            a1.errorbar(bin_centers, ratio_unfold, yerr=ratio_unfold_errors, fmt='none', linewidth=2, markersize=4, capsize=6, color='C1')
            a1.hist(bin_centers, bins=bins, weights=ratio_reco, histtype="step", label='detector/truth',linewidth=3.0, linestyle="dashed")

            # Major ticks every 20, minor ticks every 5
            major_ticks = np.arange(0.6, 1.45, 0.2)
            minor_ticks = np.arange(0.6,1.45, 0.1)

            a1.set_yticks(major_ticks)
            a1.set_yticks(minor_ticks, minor=True)

            # Or if you want different settings for the grids:
            a1.grid(which='minor', alpha=0.6)
            a1.grid(which='major', alpha=0.9)

            a1.legend(loc='upper left')
            a1.set_ylim(0.6, 1.45)
            a1.set_ylabel(r'ratios')
            a1.set_xlabel(axis_label[i])
            a1.set_xlim(x_axis_min[i], x_axis_max[i])

            f.tight_layout()
            f.subplots_adjust(wspace=0, hspace=0.05)
            f.savefig(configs.plots_path + exp_unfold + "_histLog_" + vec[i] + '_4dist'+".png")

            plt.show()
#             print(1')


            plt.clf()
            plt.close()
        
    print(distance_record)
    dict_name = 'toy_'+configs.exparameter+'_4dist'+'.pkl'
    with open("./distance_record/"+dict_name, "wb") as pickle_file:
        pickle.dump(distance_record, pickle_file)
        
else:
    print("train data type: ", configs.train_type)
    for dataset_name in ['lepqua_NNPDF23lo0130', 'ttbar_CT14lo_vincia', 'wjets_CT14lo', 'zjets_CTEQ6L1']:
#     for dataset_name in ['wjets_CT14lo_part']:
#     print("test data type: ", configs.exp_print)


    ## import data from output files



    # reco = np.load(configs.input_path + "reco_" + configs.exp_print + ".npy")
    # truth = np.load(configs.input_path + "truth_" + configs.exp_print + ".npy")

        if k!=-1:
            kstr = 'k'+str(k)
        else:
            kstr = ''
        reco = np.load(configs.input_path + "reco_" + dataset_name+".npy", mmap_mode='r')[:,:]
        truth = np.load(configs.input_path + "truth_" + dataset_name +".npy", mmap_mode='r')[:,:]
        unfold = np.load("./outputs/" + '_real25_batchsize_FM' + "/unfold_" + dataset_name +kstr+ ".npy")
        print(("./outputs/" + '_real25_batchsize_FM' + "/unfold_" + dataset_name +kstr+ ".npy"))

        vaild_ind = (~np.isnan(unfold[:,0]))& (np.abs(unfold[:,0]) <= 1e4)
        unfold = unfold[vaild_ind]
        reco = reco[vaild_ind]
        truth = truth[vaild_ind]
        # unfold = np.load("./outputs/" + 'moments_condon_y1_res' +"/unfold_" + dataset_name + ".npy")

        unfold = unfold[:,:configs.data_dim]#*configs.norm_vec
        truth = truth[:,:configs.data_dim]
        reco = reco[:,:configs.data_dim]


        n_unf = unfold.shape[0]
        print(n_unf)

        # if not unfolding full data distribution
        reco = reco[:n_unf,:]
        truth = truth[:n_unf,:]

        ## undo normalization
#         reco = reco*configs.norm_vec
#         truth = truth*configs.norm_vec

        x_text_1 = 0.10
        x_text_2 = 0.40
        y_text_1 = 0.92

        # ranges for plotting
        n_bins = 51

        plot_pT_min = 30
        plot_pT_max = 800
        if configs.exp_scale < 0.1:
            plot_pT_max = 600
        plot_E_min = 0
        plot_E_max = 2000
        plot_eta_min = -4.4
        plot_eta_max = 4.4
        plot_phi_min = -3.2
        plot_phi_max = 3.2

        x_axis_min = [plot_pT_min, plot_eta_min, plot_phi_min, plot_E_min, plot_pT_min, plot_pT_min, plot_pT_min]
        x_axis_max = [plot_pT_max, plot_eta_max, plot_phi_max, plot_E_max, plot_pT_max, plot_pT_max, plot_pT_max]
        y_axis_space = [50, 10, 2, 10, 10, 10, 10]

        ## make plots!!!

        vec = ["pT", "eta", "phi", "E", 'px', 'py','pz']
        axis_label = [r'$p_T$ [GeV]', r'$\eta$', r'$\phi$', r'$E$ [GeV]', r'$p_x$', r'$p_y$', r'$p_z$']
        for i in range(len(vec)):
            if i == 0: # pT
                bins = np.array([0, 60, 80, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000])
            elif i == 1: # eta
                bins = np.linspace(plot_eta_min, plot_eta_max, n_bins)
            elif i == 2: # phi
                bins = np.linspace(plot_phi_min, plot_phi_max, n_bins)
            elif i == 3: # E
                bins = np.array(
                    [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
                     1850, 2000, 2250, 2500, 3000, 4000])
            else:
                bins = np.array([0, 60, 80, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000])
            # if i == 4 or i == 5:
            #     bins = np.linspace(-0.5, 0.5, n_bins)
            # if i == 6:
            #     bins = np.linspace(-2, 2, n_bins)

            bin_centers = 0.5*(bins[1:] + bins[:-1])
            print(truth[:,5])
            # c = ppp
            
            if remove_outliers:
                ## get rid of huge outliers
                mask = np.abs(unfold[:,i]) < 100000
                wass_unfold = wasserstein_distance(truth[:,i][mask], unfold[:,i][mask])
                wass_reco = wasserstein_distance(truth[:,i][mask], reco[:,i][mask])
                mse_unfold = np.linalg.norm(truth[:,i][mask] - unfold[:,i][mask]) ** 2/ mask.shape[0]
                mse_reco = np.linalg.norm(truth[:,i][mask] - reco[:,i][mask]) ** 2/ mask.shape[0]
            else:
                wass_unfold = wasserstein_distance(truth[:,i], unfold[:,i])
                wass_reco = wasserstein_distance(truth[:,i], reco[:,i])
                mse_unfold = np.linalg.norm(truth[:,i] - unfold[:,i]) ** 2/ truth.shape[0]
                mse_reco = np.linalg.norm(truth[:,i] - reco[:,i]) ** 2/ truth.shape[0]
                
            distance_record[dataset_name+ vec[i]+'unfold'] = wass_unfold
            distance_record[dataset_name + vec[i]+'reco'] = wass_reco
            mse_record[dataset_name+ vec[i]+'unfold'] = mse_unfold
            mse_record[dataset_name + vec[i]+'reco'] = mse_reco

            plt.clf()

            f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(6, 6))


            # Calculate histograms and errors
            t, _ = np.histogram(truth[:,i], bins=bins, density=False)
            p, _ = np.histogram(unfold[:,i], bins=bins, density=False)
            r, _ = np.histogram(reco[:,i], bins=bins, density=False)

            t_errors = calculate_errors(t, bins)

            # plot truth, reco, and unfold
            a0.hist(truth[:,i], bins=bins, density=True, histtype='step', label='truth',linewidth=3.5, zorder=1)
            a0.hist(unfold[:,i], bins=bins, density=True, histtype='step', label='unfold', linewidth=3, linestyle="dashed", zorder=2)
            a0.hist(reco[:,i], bins=bins, density=True, histtype='step', label='detector', linewidth=3, linestyle="dashed", zorder=3)


            a0.legend(loc='upper right')

            a0.set_xticklabels([])
            a0.set_xlim(x_axis_min[i], x_axis_max[i])
            a0.set_yscale('log')
            a0.set_ylabel(r'$d\sigma/dx$')
            # set extra space for text
            y_min, y_max = a0.get_ylim()
            a0.set_ylim(y_min, y_max*y_axis_space[i])
            if vec[i] == "phi":
              a0.set_ylim(0.05, y_max*y_axis_space[i])

            # add text with metrics 
            a0.text(x_text_1, y_text_1+0.02, r'Wasserstein:', transform=a0.transAxes) 
            a0.text(x_text_1, y_text_1-0.05, r'  unfolded = %.2f' % wass_unfold, transform=a0.transAxes)
            a0.text(x_text_1, y_text_1-0.12, r'  detector = %.2f' % wass_reco, transform=a0.transAxes)
            # print the process name of the unfolded distribution
            a0.text(0.10, 0.06, dataset_name, transform=a0.transAxes)




            ## make array of ones like bin_centers
            ones = []
            for j in range(len(bin_centers)):
                ones.append(1)

            ratio_unfold = p / t
            ratio_reco = r / t

            ratio_unfold_errors = ratio_unfold * t_errors
            # ratio_unfold_errors = ratio_unfold * np.sqrt((t_errors)**2 + (calculate_errors(p, bins))**2)


            a1.hist(bin_centers, bins=bins, weights=None, histtype="step", label='',linewidth=2.0, alpha=0.8)
            a1.hist(bin_centers, bins=bins, weights=ratio_unfold, histtype="step", label='unfold/truth',linewidth=3.0)
            a1.errorbar(bin_centers, ratio_unfold, yerr=ratio_unfold_errors, fmt='none', linewidth=2, markersize=4, capsize=6, color='C1')
            a1.hist(bin_centers, bins=bins, weights=ratio_reco, histtype="step", label='detector/truth',linewidth=3.0, linestyle="dashed")

            # Major ticks every 20, minor ticks every 5
            major_ticks = np.arange(0.6, 1.45, 0.2)
            minor_ticks = np.arange(0.6,1.45, 0.1)

            a1.set_yticks(major_ticks)
            a1.set_yticks(minor_ticks, minor=True)

            # Or if you want different settings for the grids:
            a1.grid(which='minor', alpha=0.6)
            a1.grid(which='major', alpha=0.9)

            a1.legend(loc='upper left')
            a1.set_ylim(0.6, 1.45)
            a1.set_ylabel(r'ratios')
            a1.set_xlabel(axis_label[i])
            a1.set_xlim(x_axis_min[i], x_axis_max[i])

            f.tight_layout()
            f.subplots_adjust(wspace=0, hspace=0.05)
            f.savefig(configs.plots_path + dataset_name + vec[i]+configs.infonet_type +kstr+".png")
            plt.show()


            plt.clf()
            plt.close()
        
        
    print(distance_record)
    print(mse_record)
    dict_name = 'distance_record_res'+kstr+'.pkl'
    with open(configs.plots_path+dict_name, "wb") as pickle_file:
        pickle.dump(distance_record, pickle_file)
        pickle.dump(mse_record, pickle_file)

        

