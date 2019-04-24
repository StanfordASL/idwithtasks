#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import seaborn
seaborn.set()

plt.rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
plt.rc('font', **font)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

# was 34 and 18 previously

FONT_SIZE = 34
LEGEND_FONT_SIZE = 16

XTICK_LABEL_SIZE = 16
YTICK_LABEL_SIZE = 16

import matplotlib.pylab as pylab
params = {'legend.fontsize': LEGEND_FONT_SIZE,
         'axes.labelsize': FONT_SIZE,
         'axes.titlesize': FONT_SIZE,
         'xtick.labelsize': XTICK_LABEL_SIZE,
         'ytick.labelsize': YTICK_LABEL_SIZE,
         'figure.autolayout': True}
pylab.rcParams.update(params)




def k_bound_latex_name(model_name = None, title = False, camera_ready = True):

    if model_name[0] == 'k' and model_name !='k-bound':
        if title:
            k_bound_latex = r"$z_{" + str(model_name[1]) + "}$"

            if camera_ready:
                #k_bound_latex = r"$\mathrm{surveil}_{z_{" + str(model_name[1]) + "}}$"
                k_bound_latex = r"$\texttt{surveil}_{z_{" + str(model_name[1]) + "}}$"

        else:
            k_bound_latex = r"$z_{" + str(model_name[1]) + "}-\mathrm{bound}$"

            if camera_ready:
                #k_bound_latex = r"$\mathrm{surveil}_{z_{" + str(model_name[1]) + "}}$"
                k_bound_latex = r"$\texttt{surveil}_{z_{" + str(model_name[1]) + "}}$"

    elif model_name == 'k-bound':
        k_bound_latex = r"$z_{0}-\mathrm{bound}$"

        if camera_ready:
            #k_bound_latex = r"$\mathrm{surveil}_{z_{0}}$"
            k_bound_latex = r"$\texttt{surveil}_{z_{0}}$"
    else:
        k_bound_latex = model_name

    return k_bound_latex

"""
assumes 3 models

"""
def plot_belief_history(B_hist = None, reward_params_dict = None, true_model = None, plot_file = None, model_names = None, ylabel = r'Belief $B_k$', linewidth=3.0, xlim = None, ylim = None):

    B_hist = np.asarray(B_hist)

    num_models = len(model_names)
    print(num_models)
    print(model_names)

    model_names_latex = []
    for i, model_name in enumerate(model_names):
        plt.plot(B_hist.T[i], linewidth=linewidth)
        plt.hold(True)

        model_names_latex.append(k_bound_latex_name(model_name = model_name))

    #title_str = ' '.join(['true: ', true_model, 'control_wt: ', str(reward_params_dict['control_weight']), 'entropy_wt: ', str(reward_params_dict['entropy_weight'])])
    model_str = r"true = " + k_bound_latex_name(model_name = true_model)
    #alpha_str = r"$\alpha = " + str(reward_params_dict['control_weight']) + r"$"
    #beta_str = r"$\beta = " + str(reward_params_dict['entropy_weight']) + r"$"


    alpha_str = r"$\beta_{C} = " + str(reward_params_dict['control_weight']) + r"$"
    beta_str = r"$\beta_{I} = " + str(reward_params_dict['entropy_weight']) + r"$"

    title_str = ' , '.join([model_str, alpha_str, beta_str])

    plt.xlabel(r'Iteration $k$')


    plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    plt.legend(model_names_latex)
    plt.title(title_str)
    plt.savefig(plot_file)

    plt.close()


def plot_ts(ts_vector = None, reward_params_dict = None, true_model = None, plot_file = None, ylabel = None, lw=3.0, xlim = None, ylim = None, legend = None):

    plt.plot(ts_vector, lw=lw)

    #title_str = ' '.join(['true: ', true_model, 'control_wt: ', str(reward_params_dict['control_weight']), 'entropy_wt: ', str(reward_params_dict['entropy_weight'])])

    model_str = r"true = " + true_model

    alpha_str = r"$\alpha = " + str(reward_params_dict['control_weight']) + r"$"

    beta_str = r"$\beta = " + str(reward_params_dict['entropy_weight']) + r"$"

    title_str = ' , '.join([model_str, alpha_str, beta_str])

    plt.xlabel('iterations')
    plt.ylabel(ylabel)

    if legend:
        plt.legend(legend)

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    plt.title(title_str)
    plt.savefig(plot_file)

    plt.close()

def basic_scatterplot(ts_x = None, ts_y = None, title_str = None, plot_file = None, ylabel = None, lw=3.0, ylim = None, xlabel = 'time', xlim = None):

    plt.scatter(ts_x, ts_y, lw=lw)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()


def basic_plot_ts(ts_vector = None, title_str = None, plot_file = None, ylabel = None, lw=3.0, ylim = None, xlabel = 'time', xvec = None):

    if xvec != None:
        plt.plot(xvec, ts_vector, lw=lw)
    else:
        plt.plot(ts_vector, lw=lw)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim:
        plt.ylim(ylim[0], ylim[1])

    plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()


def overlaid_ts(normalized_ts_dict = None, title_str = None, plot_file = None, ylabel = None, lw=3.0, xlabel = 'time', style_dict = None, fontsize = 30, xticks = None, xvec = None):

    for ts_name, ts_vector in normalized_ts_dict.iteritems():

        if style_dict:
            if xvec != None:
                plt.plot(xvec, ts_vector, lw=lw, label = ts_name, ls = style_dict[ts_name])
            else:
                plt.plot(ts_vector, lw=lw, label = ts_name, ls = style_dict[ts_name])

        else:
            if xvec != None:
                plt.plot(xvec, ts_vector, lw=lw, label = ts_name)
            else:
                plt.plot(ts_vector, lw=lw, label = ts_name)

        plt.hold(True)

    if fontsize:
        if xlabel:
            plt.xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            plt.ylabel(ylabel, fontsize=fontsize)
    else:
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

    if xticks:
        plt.xticks(xticks)

    plt.legend(loc='best')

    plt.title(title_str, fontsize=fontsize)
    plt.savefig(plot_file)
    plt.close()

# plot grid KPI subfigures
def plot_grid(normalized_ts_dict=None,
                    title_str = None,
                    plot_file=None,
                    lw = 3.0,
                    xlabel = None):

    nrow = len(normalized_ts_dict.keys())
    ncol = 1

    plt.close('all')
    f, axarr = plt.subplots(nrow, 1, sharex=True)

    if title_str:
        plt.title(title_str)
    #print(axarr)
    #print(axarr[0])

    row = 0
    for ylabel_name, timeseries_dict in normalized_ts_dict.iteritems():

        if 'x' in timeseries_dict.keys():
            axarr[row].plot(timeseries_dict['x'], timeseries_dict['ts_vector'], lw = lw)
        else:
            axarr[row].plot(timeseries_dict['ts_vector'], lw = lw)

        axarr[row].set_ylabel(ylabel_name)

        if timeseries_dict['ylim']:
            axarr[row].set_ylim(timeseries_dict['ylim'])

        if timeseries_dict['xlim']:
            axarr[row].set_xlim(timeseries_dict['xlim'])

        if 'yticks' in timeseries_dict.keys():
            if timeseries_dict['yticks']:
                axarr[row].set_yticks(timeseries_dict['yticks'])

        row+= 1

    if xlabel:
        plt.xlabel(xlabel)
    plt.show()
    plt.savefig(plot_file)
    plt.close()


def plot_cdf(data_vector = None, xlabel = None, plot_file = None, title_str = None):

    np_data = np.array(data_vector)

    clean_data = np_data[~np.isnan(np_data)]

    sns.distplot(clean_data)

    plt.xlabel(xlabel)

    plt.title(title_str)

    plt.savefig(plot_file)
    plt.close()


def plot_several_cdf(data_vector_list = None, xlabel = None, plot_file = None, title_str = None, legend = None):

    for i, data_vector in enumerate(data_vector_list):
        sns.distplot(data_vector)
        plt.hold(True)

    plt.xlabel(xlabel)
    plt.title(title_str)
    plt.legend(legend)

    plt.savefig(plot_file)
    plt.close()


def plot_grouped_boxplot(df = None, x_var = None, y_var = None, plot_file = None, ylim = None, title_str = None, order_list = None, pal = None):

    fig = plt.figure()

    if not pal:
        if order_list:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list)
        else:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list)

    if pal:
        if order_list:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list, palette = pal)
        else:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list, palette = pal)


    if ylim:
        plt.ylim(ylim[0], ylim[1])
    #sns.plt.tight_layout()
    #sns.plt.savefig(plot_file)
    #sns.plt.clf()

    if title_str:
        plt.title(title_str)

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.clf()
    plt.close()



if __name__ == "__main__":

    x = np.random.normal(size=1000)

    y = np.random.normal(loc = 2, scale = 0.1, size=1000)

    z = np.random.normal(loc = 5, scale = 0.1, size=1000)

    plot_file = 'cdf.pdf'
    xlabel = 'distance'

    plot_cdf(data_vector = x, xlabel = xlabel, plot_file = plot_file, title_str = 'Cart-Cart')

    plot_file = 'several_cdf.pdf'

    data_vector_list = [x, y, z]
    legend_vec = ['x', 'y', 'z']

    plot_several_cdf(data_vector_list = data_vector_list, xlabel = xlabel, plot_file = plot_file, title_str = 'Cart-Cart', legend = legend_vec)
