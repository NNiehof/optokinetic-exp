"""
Plot Figure 3 for optokinetic paper
"""

import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from psychometric_curve_fit import *
from fit_sinusoid import plot_sinusoid_data, fit_sine_fixed_freq

debug = False

# read in dataframe
df = pd.read_pickle('U:\Documents\Optokinetic experiment\optokinetic_data_all_sj.pkl')
subjects = sorted(df['sj_id'].unique())
conditions = ['CW', 'NONE', 'CCW']
frame_angles = ['-45', '-33.75', '-22.5', '-11.25', '0', '11.25', '22.5', '33.75', 'noframe']
stimRange = np.arange(-30, 30, 0.5)
curves_for_plotting = ["-22.5", "0", "22.5"]

if debug:
    subjects = [0, 1]

# colours for plotting
colours = ["black", "black", "indigo", "black", "green", "black", "orangered", "black", "black"]

pse = {cond: np.empty([len(subjects), len(frame_angles)]) for cond in conditions}
std_frame = {cond: np.empty([len(subjects), len(frame_angles)]) for cond in conditions}
noframe_pse = {cond: np.empty(len(subjects)) for cond in conditions}
noframe_std = {cond: np.empty(len(subjects)) for cond in conditions}
mean_frames_pse = {}
amplitudes = {}
phases = {}
sine_rmse = {}
rmse_list = []

# subplot indices
grid = gridspec.GridSpec(3, 3)

for sj in subjects:
    mean_frames_pse[sj] = {}
    amplitudes[sj] = {}
    phases[sj] = {}
    sine_rmse[sj] = {}

    fig1 = plt.figure()

    for i_cond, cond in enumerate(conditions):
        pselist = []
        stdlist = []

        # make figure axes to plot all psychometric curves and PSEs in one figure
        # ax_psy = plt.subplot(3, 2, psy_ind[i_cond])
        # ax_pse = plt.subplot(3, 2, pse_ind[i_cond])
        ax_psy = plt.subplot(grid[i_cond, :-1])
        ax_pse = plt.subplot(grid[i_cond, -1])

        for i_frame, frame_ang in enumerate(frame_angles):
            # slice data subset
            selection = pd.DataFrame()
            selection['sj_id'] = df['sj_id'] == sj
            selection['dotsRotation'] = df['dotsRotation'] == cond
            selection['frameAngle'] = df['frameAngle'] == frame_ang
            sdata = df[selection.all(1)]

            # fit sigmoid to data
            stims = map(float, sdata['rodAngle'])
            resps = map(float, sdata['response'])
            xydata = success_ratio(stims, resps)
            psy, psy_params = fit_sigmoid(xydata, xdata_range=stimRange, noLapses=True)

            col = colours[i_frame]
            # plot response ratios and psychometric curves
            if frame_ang in curves_for_plotting:
                plot_psychometric_curve(ax_psy, stimRange, psy, xydata[0], xydata[1], col)

            # collect PSE and std of each psychometric function
            if frame_ang == 'noframe':
                noframe_pse[cond][sj] = psy_params[0]
                noframe_std[cond][sj] = 1/psy_params[1]
            else:
                pselist.append(psy_params[0])
                stdlist.append(1/psy_params[1])
        pselist.append(pselist[0])
        pse[cond][sj, :] = pselist
        std_frame[cond][sj, 0:-1] = stdlist
        std_frame[cond][sj, -1] = stdlist[0]
        mean_frames_pse[sj][cond] = np.mean(pselist)

        # format plots
        ax_psy.set_title(cond)
        ax_psy.set_ylim([-0.1, 1.1])
        ax_psy.set_xlim([-17, 17])
        # ax_psy.set_xticks([-10, -5, 0, 5, 10])
        if cond == 'NONE':
            ax_psy.tick_params(axis='both', which='both', left='on', labelleft='on')
            ax_psy.set_ylabel('Proportion \'CW\' responses')
        if cond == 'CCW':
            ax_psy.set_xlabel('Line orientation (deg)')

        # plot PSE
        frames = [-45, -33.75, -22.5, -11.25, 0, 11.25, 22.5, 33.75, 45]
        for frame, mu, colour in zip(frames, pselist, colours):
            ax_pse.plot(frame, mu, "o", color=colour)
            # ax_pse.plot(frame, mu, "o", markeredgecolor=colour, markerfacecolor="None")

        ax_pse.set_ylim([-6, 7])
        ax_pse.spines['top'].set_visible(False)
        ax_pse.spines['right'].set_visible(False)
        # ax_pse.set_yticks([-6, -3, 0, 3])
        ax_pse.set_title(cond)
        if cond == 'NONE':
            ax_pse.tick_params(axis='both', which='both', left='on', labelleft='on')
            ax_pse.set_ylabel('PSE (deg)')
        if cond == 'CCW':
            ax_pse.set_xlabel('Frame angle (deg)')

    plt.tight_layout()
    # fname = "sj_{0}".format((sj + 1))
    # plt.savefig(fname)
plt.show()
