# -*- coding: utf-8 -*-
"""
Fit a sine to periodic data with ``fit_site_fixed_freq`` and plot the data
and sine with ``plot_sinusoid_data``
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import leastsq


def fit_sine_fixed_freq(x, data, amp_guesses, phase_guesses, bias_guesses,
                        period):
    """
    Fit a sinusoid to data, with the period parameter fixed, and free
    parameters amplitude, phase and bias. The fit is optimised by minimising
    the sum of squares between data and sine (see scipy.optimize.leastsq).
    This is done for each combination of initial guesses, to avoid getting
    stuck in a local minimum. The set of parameters with the smallest RMSE
    is taken as the final result.

    Parameters
    ---------
    x : ndarray
        Values for the x-axis, in radians.
    data : ndarray or list
        Data to be fitted.
    amp_guesses : ndarray
        Initial guesses for the amplitude.
    phase_guesses : ndarray
        Initial guesses for the phase shift from zero.
    bias_guesses : ndarray
        Initial guesses for the bias or vertical offset.
    period : float
        Period in radians.
    
    Returns
    -------
    amp : float
        Sine amplitude.
    phaseshift : float
        Sine phase shift from zero.
    bias : float
        Sine offset, or shift on the y-axis.
    rmse : float
        Root mean squared error between the fitted function and the data.
    """
    param_list = []
    residual_list = []
    rmse_list = []
    # grid with all combinations of initial guesses
    grid = list(product(amp_guesses, phase_guesses, bias_guesses))
    # function for the least squares fit
    fit_sine =lambda b: b[0]*np.sin((2*np.pi/period)* x + b[1]) + b[2] - data
    
    for guesses in grid:
            params,cov,infodict,mesg,ier = leastsq(fit_sine, guesses,
                                                   full_output=True)
            param_list.append(params)
            residuals = infodict['fvec']
            residual_list.append(residuals)
            # find RMSE from residuals
            chisq = np.sum(residuals**2)
            dof = len(x) - len(params)
            rmse_list.append(np.sqrt(chisq/dof))
    # get optimal parameters
    best_fit = rmse_list.index(min(rmse_list))
    rmse = rmse_list[best_fit]
    amp, phaseshift, bias = param_list[best_fit]
    return amp, phaseshift, bias, rmse


def plot_sinusoid_data(x, data, amp, period, phaseshift, bias, saveplot=False,
                       fname=None, colourmap=None, ax=None):
    """
    Plot data points and sinusoid.

    Parameters
    ---------
    x : ndarray
        Values for the x-axis, in radians.
    data : ndarray or list
        Data points to be plotted.
    amp : float
        Sine amplitude.
    period : float
        Sine period.
    phaseshift : float
        Sine phase shift from zero.
    bias : float
        Sine offset, or shift on the y-axis.
    saveplot : bool, optional
        Save plot to file (fname must be defined) if True.
    fname : str, optional
        Path and name of file to save the current figure, using matplotlib.pyplot.savefig.
        File extensions can be png, pdf, ps, eps or svg. Must be defined when saveplot is set to True.
    colourmap : ndarray, optional
        Matplotlib colourmap of len(x).
    ax : axis object, optional
        Matplotlib axis handle to plot on.
    
    Notes
    -----
    Call matplotlib.pyplot.show() after the call to plot_sinusoid_data to display the figure.
    """
    x_grid = np.linspace(x[0], x[-1], 100)
    sinusoid = amp* np.sin((2*np.pi/period)* x_grid + phaseshift) + bias
    
    if ax is None:
        ax, fig = plt.figure()
        ax.set_ylabel('PSE')
        ax.set_xlabel('frame angle (deg)')
    
    for i_x, xval in enumerate(x):
        if colourmap is not None:
            col = colourmap[i_x,:]
            ax.plot(np.rad2deg(xval), data[i_x], 'o', color=col)
        else:
            ax.plot(np.rad2deg(xval), data[i_x], 'o')
    ax.plot(np.rad2deg(x_grid), sinusoid, '-', color='black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)   
    ax.set_ylim((-10,10))

    if saveplot:
        plt.savefig(fname)
