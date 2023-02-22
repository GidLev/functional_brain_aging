import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor
from scipy import stats
import pandas as pd

def bias_correction(age_train, ba_delta_train, age_test):
    '''
    Correct for regression dilution in brain age

    Beheshti, I., Nugent, S., Potvin, O., & Duchesne, S. (2019). Bias-adjustment
    in neuroimaging-based brain age frameworks: A robust scheme.
    NeuroImage: Clinical, 24, 102063.
    https://www.sciencedirect.com/science/article/pii/S2213158219304103
    '''

    slope, intercept, r_value, p_value, _ = stats.linregress(age_train, ba_delta_train)
    # print('slope = {}, intercept = {}, r_value = {}, p_value = {}'
    #       .format(slope, intercept, r_value, p_value))
    offset = (slope * age_test) + intercept
    return offset

def calc_delta(chronological_age, predicted_age):
    chronological_age = pd.to_numeric(chronological_age).values
    predicted_age = pd.to_numeric(predicted_age).values
    not_nan_loc = np.isfinite(chronological_age) & np.isfinite(predicted_age)
    offset = bias_correction(chronological_age[not_nan_loc],
                                                            predicted_age[not_nan_loc] - chronological_age[not_nan_loc],
                                                            chronological_age[not_nan_loc])
    predicted_age_cleaned = predicted_age[not_nan_loc] - offset
    delta = np.zeros_like(chronological_age)
    delta[:] = np.nan
    delta[not_nan_loc] = predicted_age_cleaned - chronological_age[not_nan_loc]
    return delta

def calc_dev_intervantion(chronological_age,predicted_age_T0, predicted_age_T18, mri_gap, get_extra = False):
    chronological_age = pd.to_numeric(chronological_age).values
    predicted_age_T0 = pd.to_numeric(predicted_age_T0).values
    predicted_age_T18 = pd.to_numeric(predicted_age_T18).values
    mri_gap = pd.to_numeric(mri_gap).values
    not_nan_loc = np.isfinite(chronological_age) & np.isfinite(predicted_age_T0) & \
                  np.isfinite(predicted_age_T18) & np.isfinite(mri_gap)
    lm = LinearRegression().fit(chronological_age[not_nan_loc,np.newaxis], predicted_age_T0[not_nan_loc])
    expected, observed = np.empty_like(chronological_age), np.empty_like(chronological_age)
    residual_dev, corrected_gap =  np.empty_like(chronological_age), np.empty_like(chronological_age)
    expected[:], observed[:], residual_dev[:], corrected_gap[:] = np.nan, np.nan, np.nan, np.nan
    observed[not_nan_loc] = predicted_age_T18[not_nan_loc]
    corrected_gap[not_nan_loc] = np.squeeze(lm.predict(chronological_age[not_nan_loc,np.newaxis] +
                                            mri_gap[not_nan_loc,np.newaxis])) - \
                                 np.squeeze(lm.predict(chronological_age[not_nan_loc,np.newaxis]))
    expected[not_nan_loc] = predicted_age_T0[not_nan_loc] + corrected_gap[not_nan_loc]
    residual_dev[not_nan_loc] = observed[not_nan_loc] - expected[not_nan_loc]
    if get_extra:
        return expected, observed, residual_dev
    else:
        return residual_dev

def table_one(df, outcomes, outcomes_labels, columns, columns_labels):
    stats_signs = [' r', ' p-value']
    columns_stats = []
    for col in columns_labels:
        for sign in stats_signs:
            columns_stats.append(col + sign)
    df_new = pd.DataFrame(index=outcomes_labels, columns = columns_stats)
    for (outcome, outcome_label) in zip(outcomes, outcomes_labels):
        for (column, column_label) in zip(columns, columns_labels):
            non_nan = (df[outcome].notna()) & (df[column].notna())
            r, p = stats.pearsonr(df.loc[non_nan,outcome], df.loc[non_nan,column])
            df_new.loc[outcome_label, column_label + stats_signs[0]] = r
            df_new.loc[outcome_label, column_label + stats_signs[1]] = p
    return df_new