import numpy as np
np.random.seed(0)
from sklearn.svm import SVR
from scipy import stats
import pandas as pd
import plot_utils as putils

plot_path = '/path/to/plots'
data_path = '/path/to/data'
permut_test, permut_n = True, 1000

# load the DIRECT-PLUS data
direct_plus_data = np.load('/path/to/direct_plus_data.npy')
direct_plus_func_matrices_T0 = direct_plus_data['func_matrices_T0']
direct_plus_func_matrices_T18 = direct_plus_data['func_matrices_T18']
direct_plus_ages_T0 = direct_plus_data['ages_T0']
direct_plus_ages_T18 = direct_plus_data['ages_T18']
sno_list = direct_plus_data['sno_list']

# load the NKI FC matrices
nki_data = np.load('/path/to/nki_data.npy')
nki_func_matrices = nki_data['func_matrices']
nki_ages = nki_data['ages']

# load the camcan FC matrices
camcan_data = np.load('/path/to/camcan_data.npy')
camcan_func_matrices = camcan_data['func_matrices']
camcan_ages = camcan_data['ages']

lower_trin_mask = np.zeros((nki_func_matrices.shape[1], nki_func_matrices.shape[2]), dtype=bool)
lower_trin_mask[np.tril_indices(nki_func_matrices.shape[1], -1)] = 1

# filter the data to the relevant ages
age_min, age_max = direct_plus_ages_T0.min(), direct_plus_ages_T0.max()
relevant_ages = np.where((nki_ages >= age_min) & (nki_ages <= age_max))[0]
nki_ages = nki_ages[relevant_ages]
nki_func_matrices = nki_func_matrices[relevant_ages,...]

relevant_ages = np.where((camcan_ages >= age_min) & (camcan_ages <= age_max))[0]
camcan_ages = camcan_ages[relevant_ages]
camcan_func_matrices = camcan_func_matrices[relevant_ages,...]

# extract the lower triangle of the matrices
for mat in [direct_plus_func_matrices_T0, direct_plus_func_matrices_T18, nki_func_matrices, camcan_func_matrices]:
    mat = mat.reshape(mat.shape[0], mat.shape[1] * mat.shape[2])[:,lower_trin_mask.flatten()]

# train the model on the NKI data
model_age = SVR(kernel='linear', C=1, epsilon=0.1)  # sklearn.__version__ -> 0.23.1
n = len(nki_ages)
y_lim = (36,80) # the y-axis limits for all plots
y_ticks = [int(x) for x in range(40,81, 10)]
putils.kfold_eval(model_age, nki_func_matrices, nki_ages, k=5, permut_test=permut_test,
                  title='Validation set (NKI; n=' + str(n) + ')', ylabel='Predicted brain age (years)',
                  plot_path=plot_path + '/NKI_prediction_acc_pre_bias_corr.png', permut_n=permut_n,
                  y_lim=y_lim, y_ticks=y_ticks)

model_age.fit(nki_func_matrices, nki_ages)
coef_flat = model_age.coef_
nki_predicted_ages = model_age.predict(nki_func_matrices).squeeze()
camcan_predicted_ages = model_age.predict(camcan_func_matrices).squeeze()
y_predicted_t0 = model_age.predict(direct_plus_func_matrices_T0).squeeze()
y_predicted_t18 = model_age.predict(direct_plus_func_matrices_T18).squeeze()
if permut_test:
    camcan_predicted_ages_permuts = np.zeros((len(camcan_predicted_ages), permut_n))
    y_predicted_t0_permuts = np.zeros((len(y_predicted_t0), permut_n))
    y_predicted_t18_permuts = np.zeros((len(y_predicted_t18), permut_n))
    nki_ages_permuts = nki_ages.copy()
    for i in range(permut_n):
        np.random.shuffle(nki_ages_permuts)
        model_age.fit(nki_func_matrices, nki_ages_permuts)
        camcan_predicted_ages_permuts[:, i] = model_age.predict(camcan_func_matrices).squeeze()
        y_predicted_t0_permuts[:, i] = model_age.predict(direct_plus_func_matrices_T0).squeeze()
        y_predicted_t18_permuts[:, i] = model_age.predict(direct_plus_func_matrices_T18).squeeze()

r0, p0 = stats.pearsonr(y_predicted_t0, direct_plus_ages_T0)
r18, p18 =  stats.pearsonr(y_predicted_t18, direct_plus_ages_T18)
print('T0 r= {:.3f}, p = {:.3f}\n T18 r = {:.3f}, p = {:.3f}'.
             format(r0, p0, r18, p18))

# save the results of the prediction
data = {'sno': sno_list,
        'func_age_pred_t0': y_predicted_t0,
        'func_age_pred_t18': y_predicted_t18,
        'age_t0': direct_plus_ages_T0,
        'age_t18': direct_plus_ages_T18}
pd.DataFrame(data).to_csv(data_path + 'predicted_ages.csv')

## plot the prediction accuracy

# prediction accuracy on the DIRECT-PLUS data
mae_p = None
n = len(direct_plus_ages_T0)
if permut_test:
    MAE = np.mean(np.abs(direct_plus_ages_T0 - y_predicted_t0))
    MAE_null = np.mean(np.abs(direct_plus_ages_T0[:,np.newaxis] - y_predicted_t0_permuts),axis=0)
    mae_p = (MAE_null < MAE).mean()
putils.reg_plot_paper(direct_plus_ages_T0, y_predicted_t0, xlabel='Baseline chronological age (years)',
                      ylabel='Predicted brain age (years)', title='DIRECT-PLUS cohort (n=' + str(n) + ')',
                      type='pearson', save_path=plot_path + '/TO_prediction_acc_pre_bias_corr.png',
                      mae=True, mae_p = mae_p, y_lim=y_lim, y_ticks=y_ticks)
if permut_test:
    MAE = np.mean(np.abs(direct_plus_ages_T18 - y_predicted_t18))
    MAE_null = np.mean(np.abs((direct_plus_ages_T18[:,np.newaxis]) - y_predicted_t18_permuts),axis=0)
    mae_p = (MAE_null < MAE).mean()
putils.reg_plot_paper(direct_plus_ages_T18, y_predicted_t18, xlabel='T18 chronological age (years)',
                      ylabel='Predicted brain age (years)', title='DIRECT-PLUS cohort (T18; n=' + str(n) + ')',
                      type='pearson', save_path=plot_path + '/T18_prediction_acc_pre_bias_corr.png',
                      mae=True, mae_p = mae_p, y_lim=y_lim, y_ticks=y_ticks)

# prediction accuracy on the Cam-CAN data
n = int(camcan_ages)
if permut_test:
    MAE = np.mean(np.abs(camcan_ages - camcan_predicted_ages))
    MAE_null = np.mean(np.abs(camcan_ages[:, np.newaxis] - camcan_predicted_ages_permuts), axis=0)
    mae_p = (MAE_null < MAE).mean()
putils.reg_plot_paper(camcan_ages, camcan_predicted_ages, xlabel='Chronological age (years)',
                      ylabel='Predicted brain age (years)', title='Test set (Cam-CAN; n=' + str(n) + ')',
                      type='pearson', save_path=plot_path + '/camcan_prediction_acc.png',
                      mae=True, mae_p = mae_p, y_lim=y_lim, y_ticks=y_ticks)


