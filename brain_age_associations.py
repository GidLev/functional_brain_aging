import numpy as np
from scipy import stats
from matplotlib import pylab as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
from brain_age_utils import table_one
import plot_utils as putils

# load the data
plot_path = '/path/to/plots'
data_path = '/path/to/data'

# load the DIRECT-PLUS data
brain_age_df = pd.read_csv(data_path + 'brain_age_measures.csv', index_col='sno')
ffq_food_df = pd.read_csv(data_path + '/FFQ.csv', index_col='sno')
clinical_outcomes_df = pd.read_csv(data_path + '/clinical_outcomes.csv', index_col='sno')
food_vars = ['sweets_and_beverages_change18', 'weekly_mankai_18',
             'nuts_seeds_change18', 'egg_milk_change18', 'beef_p_change18',
             'processed_food_change18',
             'green_tea_filled_change18', 'walnuts_change18']
ffq_food_df = ffq_food_df.loc[:,food_vars]

main_df =  pd.concat([brain_age_df, ffq_food_df, clinical_outcomes_df], axis=1)

# Generate table 1

t0_obesity = ['BMI_0','WC_0']
t0_liver = ['AST_0', 'ALT_0', 'GGT_0', 'ALKP_0', 'FGF21_0', 'Chemerin_0']
t0_glyc = ['Glucose_0', 'HOMAIR_0', 'HbA1c_0']
t0_lipids = ['Cholesterol_0','HDLc_0', 'LDLc_0', 'Triglycerides_0']
t0_img = ['TI_HFF_0', 'vat_0','ssc_0', 'dsc_0']
t0_outcomes = t0_obesity + t0_liver + t0_glyc + t0_lipids + t0_img

labels = ['BMI (kg/m²)', 'WC (cm)', 'AST (U/L)', 'ALT (U/L)', 'GGT (U/L)',
          'ALKP (mg/dL)', 'FGF 21 (pg/mL)', 'Chemerin (ng/mL)', 'Glucose (mg/dL)',
          'HOMA IR', 'HbA1c (%)', 'Cholesterol (mg/dL)', 'HDL-C (mg/dL)',
          'LDL-C (mg/dL)', 'Triglycerides (mg/dL)', 'Liver fat (cm²)',  'VAT (cm²)',
          'SSC (cm²)', 'DSC (cm²)']

table_1 = table_one(main_df, t0_outcomes + ['HOC_mean_time0'], labels + ['HOC'],
                    ['age', 'func_age_pred_t0', 'func_age_pred_t0_delta'],
                    ['Age', 'Brain age', 'T0 brain age deviation'])
table_1.to_csv('Table_1.csv')

# figure 4 - multiple scatter plots
del_18_obesity = ['BMI_del18', 'WC_del18']
del_18_liver = ['AST_del18', 'ALT_del18', 'GGT_del18', 'ALKP_del18', 'FGF21_del18', 'Chemerin_del18']
del_18_glyc = ['Glucose_del18', 'HOMAIR_del18', 'HbA1c_del18'] 
del_18_lipids = ['cholesterol_del18','HDLc_del18', 'LDLc_del18', 'Triglycerides_del18']
del_18_img = ['TI_HFF_del18', 'vat_del18','ssc_del18', 'dsc_del18']
del18_outcomes = del_18_obesity + del_18_liver + del_18_glyc + del_18_lipids + del_18_img

labels_delta = [r'$\Delta$' + x for x in labels]

colors = ['tab:blue'] * 2 + ['tab:orange'] * 6 + ['tab:brown'] * 3  + ['tab:red'] * 4 + \
         ['tab:green'] * 4

fig = plt.figure(figsize=(14, 13.5))
n_row, n_col = 4, 5
rs, ps = np.zeros(len(del18_outcomes)), np.zeros(len(del18_outcomes))
ps_corrected = np.zeros(len(del18_outcomes))
for i, measure in enumerate(del18_outcomes):
    vec_var1 = pd.to_numeric(main_df['func_age_pred_dev_intervantion']).values
    vec_var2 = pd.to_numeric(main_df[measure]).values
    filter_subjects = np.isfinite(vec_var1) & np.isfinite(vec_var2)
    vec_var1 = vec_var1[filter_subjects]
    vec_var2 = vec_var2[filter_subjects]
    n = filter_subjects.sum()
    rs[i], ps[i] = stats.pearsonr(vec_var1, vec_var2)
for color in colors:
    group_inds = np.array(colors) == color
    _, ps_corrected[group_inds] = fdrcorrection(ps[group_inds])
for i, (measure, label, color) in enumerate(zip(
        del18_outcomes, labels_delta, colors)):
    vec_var1 = pd.to_numeric(main_df['func_age_pred_dev_intervantion']).values
    vec_var2 = pd.to_numeric(main_df[measure]).values
    filter_subjects = np.isfinite(vec_var1) & np.isfinite(vec_var2)
    vec_var1 = vec_var1[filter_subjects]
    vec_var2 = vec_var2[filter_subjects]
    n = filter_subjects.sum()
    ax = fig.add_subplot(n_col, n_row, i + 1)
    sns.regplot(x=vec_var2, y=vec_var1, ax=ax, color=color,
                scatter_kws={'alpha': 0.3, 's': 25})
    if (i % n_row) == 0:
        plt.ylabel('Brain age attenuation', fontsize=15)
    else:
        plt.ylabel('')
    plt.xlabel(label, fontsize=15)
    range = vec_var2.max() - vec_var2.min()
    plt.xlim(vec_var2.min() - range * 0.05, vec_var2.max() + range * 0.05)
    ax.tick_params(labelsize=11)
    if ps[i] < 0.001:
        txt =  "r={:.3f}, ".format(rs[i]) + "p<0.001"
    else:
        txt = "r={:.3f}, ".format(rs[i]) + "p={:.3f}".format(ps[i])
    if ps_corrected[i] < 0.001:
        txt = txt + '***'
    elif ps_corrected[i] < 0.01:
        txt = txt + '**'
    elif ps_corrected[i] < 0.05:
        txt = txt + '*'
    if ps_corrected[i] < 0.05:
        weight = 'bold'
    else:
        weight = 'normal'
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.6)
    ax.text(x=.32, y=.08, s=txt, size=11, weight=weight,
            transform=ax.transAxes, bbox=bbox_props)
plt.tight_layout()
plt.savefig(plot_path + '/func_age_pred_dev_intervantion_multiple_scatter_plots.png', dpi=600)
plt.show()

# table of all correlation + corrected by age\gender\BMI\baseline brain age
df_results = pd.DataFrame(index=labels + ['HOC'], columns=['measure', 'r_to_brain', 'p_to_brain',
                                                 'r_to_brain_c_age', 'p_to_brain_c_age',
                                                 'r_to_brain_c_gender', 'p_to_brain_c_gender',
                                                 'r_to_brain_c_brain_age_t0', 'p_to_brain_c_brain_age_t0',
                                                 'r_to_brain_c_bmi', 'p_to_brain_c_bmi'])
df_results['measure'] = del18_outcomes + ['Relative_HOC_full_Final']

brain_age = 'func_age_pred_dev_intervantion'
for i in df_results.index.values:
    df_results.loc[i, 'r_to_brain'], df_results.loc[i, 'p_to_brain'] = \
        putils.report_correlation(brain_age, df_results.loc[i, 'measure'], main_df)
    df_results.loc[i, 'r_to_brain_c_age'], df_results.loc[i, 'p_to_brain_c_age'] = \
        putils.report_partial_correlation(brain_age, df_results.loc[i, 'measure'],
                                          'age',main_df)
    df_results.loc[i, 'r_to_brain_c_bmi'], df_results.loc[i, 'p_to_brain_c_bmi'] = \
        putils.report_partial_correlation(brain_age, df_results.loc[i, 'measure'],
                                          'BMI_del18',main_df)
    df_results.loc[i, 'r_to_brain_c_brain_age_t0'], df_results.loc[i, 'p_to_brain_c_brain_age_t0'] = \
        putils.report_partial_correlation(brain_age, df_results.loc[i, 'measure'],
                                          'func_age_pred_t0',main_df)
    df_results.loc[i, 'r_to_brain_c_gender'], df_results.loc[i, 'p_to_brain_c_gender'] = \
        putils.report_partial_correlation(brain_age, df_results.loc[i, 'measure'],
                                          'gender',main_df)
df_results.to_csv(plot_path + '/correlation_of_brain_age_attun_to_clinical_measures.csv')
