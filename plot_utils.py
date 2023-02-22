import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.base import clone

def kfold_eval(model, X, y, k = 5, plot_path=None, permut_test = True, permut_n = 1000,
               title='Functional brain age accuracy', ylabel='Predicted age',
               y_lim = None, y_ticks = None):
    kf = KFold(n_splits=k)
    y_pred = np.zeros_like(y)
    for ind_kf, (train_index, test_index) in enumerate(kf.split(X)):
        curr_model = clone(model)
        curr_model.fit(X[train_index, :], y[train_index])
        y_pred[test_index] = np.squeeze(curr_model.predict(X[test_index, :]))
    if permut_test:
        MAE = np.mean(np.abs(y - y_pred))
        MAE_null = np.zeros((permut_n))
        y_permut = y.copy()
        y_pred_permut = np.zeros_like(y_pred)
        for i in range(permut_n):
            np.random.shuffle(y_permut)
            for ind_kf, (train_index, test_index) in enumerate(kf.split(X)):
                curr_model = clone(model)
                curr_model.fit(X[train_index, :], y_permut[train_index])
                y_pred_permut[test_index] = np.squeeze(curr_model.predict(X[test_index, :]))
            MAE_null[i] = np.mean(np.abs(y - y_pred_permut))
            if (i % 10) == 0:
                print('permutation {i} of {permut_n}'.format(i=i, permut_n=permut_n))
        MAE_p = (MAE_null < MAE).mean()
        if MAE_p < 0.001:
            print(title + "MAE={:.3f}, ".format(MAE) + "p={:.2e}, ".format(MAE_p))
        else:
            print(title + "MAE={:.3f}, ".format(MAE) + "p={:.3f}, ".format(MAE_p))
    else:
        MAE_p = None
    r, p = reg_plot_paper(y, y_pred, xlabel='Chronological age (years)',
                          ylabel=ylabel, title=title, mae_p = MAE_p,
                          type = 'pearson', save_path=plot_path,
                          mae = True, y_lim=y_lim, y_ticks=y_ticks)
    if p < 0.001:
        print("r={:.3f}, ".format(r) + "p={:.2e}".format(p))
    else:
        print("r={:.3f}, ".format(r) + "p={:.3f}".format(p))

def reg_plot_paper(x, y, xlabel='', ylabel='', title='', type = 'pearson', save_path=None,
                   mae = False, mae_p = None, y_lim = None, y_ticks = None):
    if type == 'pearson':
        r, p = stats.pearsonr(x, y)
    elif type == 'spearman':
        r, p = stats.spearmanr(x, y)
    print(title + ': ' + xlabel + ' to ' + ylabel + ' correlation: r= %.4f' % r + ' p= %.4f' % p)
    f, ax = plt.subplots(figsize=(5.5, 5))
    s = 25 if (len(x) > 100) else 50
    # Set up the matplotlib figure #88, 64
    if type == 'pearson':
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.3, 's':s}, color="k")
        corr_sign = 'r'
    elif type == 'spearman':
        sns.scatterplot(x=x, y=y, ax=ax, alpha = 0.3, s =s)
        evalDF = loess(x, y, 1) # get a LOESS line
        ax.plot(evalDF['x_line'], evalDF['y_line'], color='grey', linewidth=3)
        corr_sign = r'$\rho$'
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    range = x.max() - x.min()
    plt.xlim(x.min() - range * 0.05, x.max() + range * 0.05)
    if not y_lim is None:
        plt.ylim(y_lim)
    if not y_ticks is None:
        plt.yticks(y_ticks)
    ax.set_title(title, fontsize=18)
    ax.tick_params(labelsize=15)
    f.subplots_adjust(left=0.2, bottom=0.2)
    if p < 0.001:
        txt = corr_sign + "={:.3f}, ".format(r) + "p<0.001"
    else:
        txt = corr_sign + "={:.3f}, ".format(r) + "p={:.3f}".format(p)
    if mae:
        mae = np.mean(np.abs(y - x))
        if not mae_p is None:
            if mae_p < 0.001:
                txt += '\nMAE={:.3f}, '.format(mae) + "p<0.001"
            else:
                txt += '\nMAE={:.3f}, '.format(mae) + "p={:.3f}".format(mae_p)
        else:
            txt += '\nMAE={:.3f}'.format(mae)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.6)
    ax.text(x=.40, y=.05, s=txt, size=14,
                    transform=ax.transAxes, bbox=bbox_props)
    if save_path != None:
        plt.savefig(save_path, dpi=300)  #
    plt.show()
    return r, p

def loess(xvals, yvals, alpha, data = None, poly_degree=1):
    ''' LOcally-Weighted Scatterplot Smoothing - https://github.com/MikLang/Lowess_simulation/'''
    def loc_eval(x, b):
        loc_est = 0
        for i in enumerate(b): loc_est += i[1] * (x ** i[0])
        return (loc_est)
    if data == None:
        xvals, yvals = list(xvals), list(yvals)
    else:
        all_data = sorted(zip(data[xvals].tolist(), data[yvals].tolist()), key=lambda x: x[0])
        xvals, yvals = zip(*all_data)
    evalDF = pd.DataFrame(columns=['v','g'])
    n = len(xvals)
    m = n + 1
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = ((max(xvals)-min(xvals))/len(xvals))
    v_lb = min(xvals)-(.5*avg_interval)
    v_ub = (max(xvals)+(.5*avg_interval))
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)
    xcols = [np.ones_like(xvals)]
    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    X = np.vstack(xcols).T
    for i in v:
        iterpos = i[0]
        iterval = i[1]
        iterdists = sorted([(j, np.abs(j-iterval)) for j in xvals], key=lambda x: x[1])
        _, raw_dists = zip(*iterdists)
        scale_fact = raw_dists[q-1]
        scaled_dists = [(j[0],(j[1]/scale_fact)) for j in iterdists]
        weights = [(j[0],((1-np.abs(j[1]**3))**3 if j[1]<=1 else 0)) for j in scaled_dists]
        _, weights      = zip(*sorted(weights,     key=lambda x: x[0]))
        _, raw_dists    = zip(*sorted(iterdists,   key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists,key=lambda x: x[0]))
        W = np.diag(weights)
        b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ yvals)
        local_est = loc_eval(iterval, b)
        iterDF2 = pd.DataFrame({
                                'x_line'  :[iterval],
                                'y_line'  :[local_est]
                                })
        evalDF = pd.concat([evalDF, iterDF2])
    evalDF = evalDF[['x_line','y_line']]
    return(evalDF)

def report_partial_correlation(var1, var2, cov,df, title='', type = 'pearson',
                               save_path=None, verbose = True, plot = True):
    vec_var1 = pd.to_numeric(df[var1]).values
    vec_var2 = pd.to_numeric(df[var2]).values
    vec_cov = pd.to_numeric(df[cov]).values
    filter_subjects = np.isfinite(vec_var1) & np.isfinite(vec_var2) & np.isfinite(vec_cov)
    vec_var1 = vec_var1[filter_subjects]
    vec_var2 = vec_var2[filter_subjects]
    vec_cov = vec_cov[filter_subjects]
    lm = LinearRegression().fit(vec_cov[:, np.newaxis], vec_var1)
    vec_var1_res = vec_var1 - np.squeeze(lm.predict(vec_cov[:, np.newaxis]))
    lm = LinearRegression().fit(vec_cov[:, np.newaxis], vec_var2)
    vec_var2_res = vec_var2 - np.squeeze(lm.predict(vec_cov[:, np.newaxis]))
    n = filter_subjects.sum()
    if type == 'pearson':
        r, p = stats.pearsonr(vec_var1, vec_var2)
        r_partial, p_partial = stats.pearsonr(vec_var1_res, vec_var2_res)
    elif type == 'spearman':
        r, p = stats.spearmanr(vec_var1, vec_var2)
        r_partial, p_partial = stats.spearmanr(vec_var1_res, vec_var2_res)
    elif type == 'kendal':
        r, p = stats.kendalltau(vec_var1, vec_var2)
        r_partial, p_partial = stats.kendalltau(vec_var1_res, vec_var2_res)
    if verbose:
        print(title + ': ' + var1 + ' to ' + var2 + ' correlation: r= %.4f' % r +
              ' p= %.4f' % p + ' n=' + str(n) + 'partial corr:  r= %.4f' % r_partial +
              ' p= %.4f' % p_partial)
    if plot:
        f, ax = plt.subplots(figsize=(5.5, 5))
        s = 60 if (len(var1) > 100) else 120
        # Set up the matplotlib figure #88, 64
        if type == 'pearson':
            sns.regplot(x=vec_var1, y=vec_var2, ax=ax, scatter_kws={'alpha':0.3, 's':s})
            corr_sign = 'r'
        else:
            sns.scatterplot(x=vec_var1, y=vec_var2, ax=ax, alpha = 0.3, s =s)
            evalDF = loess(vec_var1, vec_var2, 1) # get a LOESS line
            ax.plot(evalDF['x_line'], evalDF['y_line'], color='grey', linewidth=3)
            if type == 'spearman':
                corr_sign = r'$\rho$'
            elif type == 'kendal':
                corr_sign = r'$\tau$'
        plt.xlabel(var1, fontsize=18)
        plt.ylabel(var2, fontsize=18)
        range = vec_var1.max() - vec_var1.min()
        plt.xlim(vec_var1.min() - range * 0.05, vec_var1.max() + range * 0.05)
        ax.set_title(title, fontsize=18)
        ax.tick_params(labelsize=15)
        f.subplots_adjust(left=0.2, bottom=0.2)
        if p < 0.001:
            txt = corr_sign + "={:.3f}, ".format(r) + "p<0.001"
        else:
            txt = corr_sign + "={:.3f}, ".format(r) + "p={:.3f}".format(p)
        if p_partial < 0.001:
            txt += '\n' + corr_sign + "(" + cov + "-corrected)={:.3f}, ".format(r_partial) + \
                   "p<0.001"
        else:
            txt += '\n' + corr_sign + "(" + cov + "-corrected)={:.3f}, ".format(r_partial) + \
                   "p={:.3f}".format(p_partial)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.6)
        ax.text(x=.05, y=.8, s=txt, size=14,
                        transform=ax.transAxes, bbox=bbox_props)
        if save_path != None:
            plt.savefig(save_path, dpi=300)  #
        plt.show()
    return r_partial, p_partial

def report_correlation(var1, var2, df, title='', type = 'pearson',
                       save_path=None, verbose = True, plot = True):
    vec_var1 = pd.to_numeric(df[var1]).values
    vec_var2 = pd.to_numeric(df[var2]).values
    filter_subjects = np.isfinite(vec_var1) & np.isfinite(vec_var2)
    vec_var1 = vec_var1[filter_subjects]
    vec_var2 = vec_var2[filter_subjects]
    n = filter_subjects.sum()
    if type == 'pearson':
        r, p = stats.pearsonr(vec_var1, vec_var2)
    elif type == 'spearman':
        r, p = stats.spearmanr(vec_var1, vec_var2)
    elif type == 'kendal':
        r, p = stats.kendalltau(vec_var1, vec_var2)
    if verbose:
        print(title + ': ' + var1 + ' to ' + var2 + ' correlation: r= %.4f' % r +
              ' p= %.4f' % p + ' n=' + str(n))
    if plot:
        f, ax = plt.subplots(figsize=(5.5, 5))
        s = 60 if (len(var1) > 100) else 120
        # Set up the matplotlib figure #88, 64
        if type == 'pearson':
            sns.regplot(x=vec_var1, y=vec_var2, ax=ax, scatter_kws={'alpha':0.3, 's':s})
            corr_sign = 'r'
        else:
            sns.scatterplot(x=vec_var1, y=vec_var2, ax=ax, alpha = 0.3, s =s)
            evalDF = loess(vec_var1, vec_var2, 1) # get a LOESS line
            ax.plot(evalDF['x_line'], evalDF['y_line'], color='grey', linewidth=3)
            if type == 'spearman':
                corr_sign = r'$\rho$'
            elif type == 'kendal':
                corr_sign = r'$\tau$'
        plt.xlabel(var1, fontsize=18)
        plt.ylabel(var2, fontsize=18)
        range = vec_var1.max() - vec_var1.min()
        plt.xlim(vec_var1.min() - range * 0.05, vec_var1.max() + range * 0.05)
        ax.set_title(title, fontsize=18)
        ax.tick_params(labelsize=15)
        f.subplots_adjust(left=0.2, bottom=0.2)
        if p < 0.001:
            txt = corr_sign + "={:.3f}, ".format(r) + "p<0.001"
        else:
            txt = corr_sign + "={:.3f}, ".format(r) + "p={:.3f}".format(p)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.6)
        ax.text(x=.05, y=.8, s=txt, size=14,
                        transform=ax.transAxes, bbox=bbox_props)
        if save_path != None:
            plt.savefig(save_path, dpi=300)  #
        plt.show()
    return r, p