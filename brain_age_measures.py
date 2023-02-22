import numpy as np
import pandas as pd
from brain_age_utils import calc_dev_intervantion, calc_delta
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

plot_path = '/path/to/plots'
data_path = '/path/to/data'

func_age_preds = pd.read_csv(data_path + 'predicted_ages.csv', index_col='sno')
subjects_mri_time = pd.read_csv('/path/to/mri_time.csv',
                                   index_col='sno').loc[:,'Time_BetweenMRI']
brain_age_df = pd.concat([func_age_preds, subjects_mri_time], axis=1)

# calc deviation metrics according to Yaskolka et al. (2021)
# Meir, A. Y., Keller, M., Bernhart, S. H., Rinott, E., Tsaban, G., Zelicha, H., ... & Shai, I. (2021).
# Lifestyle weight-loss intervention may attenuate methylation aging: the CENTRAL MRI randomized
# controlled trial. Clinical epigenetics, 13(1), 1-10.?

brain_age_df['func_age_pred_t0_delta'] = calc_delta(brain_age_df['age'], brain_age_df['func_age_pred_t0'])
brain_age_df['func_age_pred_t18_delta'] = calc_delta(pd.to_numeric(brain_age_df['age']) +
                                                     brain_age_df['Time_BetweenMRI'].values,
                                                     brain_age_df['func_age_pred_t18'])

brain_age_df['func_age_pred_expected_intervantion'], brain_age_df['func_age_pred_observed_intervantion'], \
brain_age_df['func_age_pred_dev_intervantion'] = \
    calc_dev_intervantion(brain_age_df['age'], brain_age_df['func_age_pred_t0'],
                          brain_age_df['func_age_pred_t18'], brain_age_df['Time_BetweenMRI'], get_extra = True)

brain_age_df.to_csv(data_path + 'brain_age_measures.csv')

# Comparing both age bias correction methods

df_measures = pd.DataFrame(
    data = {'Current method': brain_age_df['func_age_pred_dev_intervantion'],
            'Standard method': (brain_age_df['func_age_pred_t18_delta'] -
                                brain_age_df['func_age_pred_t0_delta'])})
r,p = stats.pearsonr(df_measures['Current method'], df_measures['Standard method'])
mae = np.abs(df_measures['Current method'] - df_measures['Standard method']).mean()
sns.scatterplot(x='Current method', y='Standard method', data=df_measures)
plt.title('Comparing both age bias correction methods (r={:.3f},p={:.2e})'.format(r,p))
plt.savefig(plot_path + '/comparing_both_age_bias_correction_methods.png', dpi=150)
plt.show()
