"""
Some first plots of individual subject data
"""

import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from psychometric_curve_fit import *
from fit_sinusoid import plot_sinusoid_data, fit_sine_fixed_freq

debug = False

# read in dataframe
df = pd.read_pickle('U:\Documents\Optokinetic experiment\optokinetic_data_all_sj.pkl')
subjects = sorted(df['sj_id'].unique())
conditions = ['CCW', 'NONE', 'CW']
frame_angles = ['-45', '-33.75', '-22.5', '-11.25', '0', '11.25', '22.5', '33.75', 'noframe']
stimRange = np.arange(-30, 30, 0.5)

if debug:
    subjects = [0, 1]

if not debug:
    # file to save the PSE per combination of rotation (CCW, CW, NONE) and frame (frame, noframe) conditions
    pse_datafile = 'U:\Documents\Optokinetic experiment\optokinetic_pse.txt'
    with open (pse_datafile, 'w') as dfile:
        dfile.write('sj_id, CCW_frame, CW_frame, NONE_frame, CCW_noframe, CW_noframe, NONE_noframe\n')
    
    # file to save fitted amplitudes, phases
    sine_datafile = 'U:\Documents\Optokinetic experiment\optokinetic_sines.txt'
    with open (sine_datafile, 'w') as dfile:
        dfile.write('sj_id, CCW_amp, CW_amp, NONE_amp, CCW_phase, CW_phase, NONE_phase\n')

# colourmap for plotting
cmap = plt.cm.plasma(np.linspace(0, 1, len(frame_angles)-1))
cmap = np.vstack((cmap, cmap[0,:])) # repeat first colour to plot x=-45 and x=45 in the same colour

pse = {cond: np.empty([len(subjects), len(frame_angles)]) for cond in conditions}
std_frame = {cond: np.empty([len(subjects), len(frame_angles)]) for cond in conditions}
noframe_pse = {cond: np.empty(len(subjects)) for cond in conditions}
noframe_std = {cond: np.empty(len(subjects)) for cond in conditions}
mean_frames_pse = {}
amplitudes = {}
phases = {}
sine_rmse = {}
rmse_list = []
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
        ax_psy = plt.subplot(3,3,(i_cond+1))
        ax_pse = plt.subplot(3,3,(i_cond+4))
        ax_std = plt.subplot(3,3,(i_cond+7))
        
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
            
            # plot response ratios and psychometric curves
            if frame_ang != 'noframe':
                col = cmap[i_frame,:]
                plot_psychometric_curve(ax_psy, stimRange, psy, xydata[0], xydata[1], col)
            
            # collect PSE and std of each psychometric function
            if frame_ang == 'noframe':
                noframe_pse[cond][sj] = psy_params[0]
                noframe_std[cond][sj] = 1/psy_params[1]
            else:
                pselist.append(psy_params[0])
                stdlist.append(1/psy_params[1])
        pse[cond][sj,0:-1] = pselist
        pse[cond][sj,-1] = pselist[0]
        std_frame[cond][sj,0:-1] =  stdlist
        std_frame[cond][sj,-1] =  stdlist[0]
        mean_frames_pse[sj][cond] = np.mean(pselist)
        
        # format plots
        ax_psy.set_title(cond)
        ax_psy.tick_params(axis='both', which='both', top='off', right='off',
                           left='on', bottom='on', labelleft='off', labelbottom='on')
        if cond == 'CCW':
            ax_psy.tick_params(axis='both', which='both', left='on', labelleft='on')
            ax_psy.set_ylabel('Proportion \'CW\' responses')
        if cond == 'NONE':
            ax_psy.set_xlabel('Line orientation (deg)')
                
        # fit sine to PSE data using arrays of initial guesses for the parameters
        frames = [-45, -33.75, -22.5, -11.25, 0, 11.25, 22.5, 33.75, 45]
        frames_rad = np.deg2rad(frames)
        pse_full_period = pselist[:]
        pse_full_period.append(pselist[0]) # -45 = 45        
        amp_guesses = np.linspace(0, 10, 5)
        phase_guesses = np.linspace(-(np.pi/4), np.pi/4, 5)
        bias_guesses = np.linspace(-10, 10, 5)
        period = np.pi/2.0 # one period goes from -45 to 45 deg
        
        # frame=45 point not included in fit because it is a duplicate of the -45 data point
        amp, phaseshift, bias, rmse = fit_sine_fixed_freq(frames_rad[0:-1], pselist, amp_guesses,
                                              phase_guesses, bias_guesses, period)
        rmse_list.append(rmse)
     
        # save sine parameters and plot sines with the data
        amplitudes[sj][cond] = amp
        phases[sj][cond] = phaseshift
        sine_rmse[sj][cond] = rmse
        plot_sinusoid_data(frames_rad, pse_full_period, amp, period,
                           phaseshift, bias, colourmap=cmap, ax=ax_pse)
        ax_pse.tick_params(axis='both', which='both', top='off', right='off',
                           left='on', bottom='on', labelleft='off', labelbottom='on')
        if cond == 'CCW':
            ax_pse.tick_params(axis='both', which='both', left='on', labelleft='on')
            ax_pse.set_ylabel('PSE (deg)')
        if cond == 'NONE':
            ax_pse.set_xlabel('Frame angle (deg)')

        # plot standard deviations from psychometric curves
        ax_std.plot(frames, std_frame[cond][sj], 'o', color='blue')
        ax_std.tick_params(axis='both', which='both', top='off', right='off',
                           left='on', bottom='on', labelleft='on', labelbottom='on')
        ax_std.set_ylim((0, 21))
        if cond == 'CCW':
            ax_std.tick_params(axis='both', which='both', left='on', labelleft='on')
            ax_std.set_ylabel('SD (deg)')
        if cond == 'NONE':
            ax_std.set_xlabel('Frame angle (deg)')

    plt.tight_layout() 
    
    if not debug:
        # write factorial PSE data to file
        formatted_pse_data = '{}, {}, {}, {}, {}, {}, {}\n'.format(
            sj, mean_frames_pse[sj]['CCW'], mean_frames_pse[sj]['CW'],
            mean_frames_pse[sj]['NONE'], noframe_pse['CCW'][sj],
            noframe_pse['CW'][sj], noframe_pse['NONE'][sj])
        with open (pse_datafile, 'a') as dfile:
            dfile.write(formatted_pse_data)
            
        # write ampltudes and phase shifts to file
        formatted_sine_data = '{}, {}, {}, {}, {}, {}, {}\n'.format(
            sj, amplitudes[sj]['CCW'], amplitudes[sj]['CW'],
            amplitudes[sj]['NONE'], phases[sj]['CCW'], phases[sj]['CW'],
            phases[sj]['NONE'])
        with open (sine_datafile, 'a') as dfile:
            dfile.write(formatted_sine_data)

if not debug:    
    # write PSE dict to file
    with open('pse_all_sj.pkl', 'wb') as fhandle:
        pickle.dump(pse, fhandle)
    with open('std_all_sj.pkl', 'wb') as fhandle:
        pickle.dump(std_frame, fhandle)
    with open('noframe_pse_all_sj.pkl', 'wb') as fhandle:
        pickle.dump(noframe_pse, fhandle)
    with open('noframe_std_all_sj.pkl', 'wb') as fhandle:
        pickle.dump(noframe_std, fhandle)
    scipy.io.savemat('pse_all_sj.mat', mdict={'pse_frame': pse,
                                              'std_frame': std_frame,
                                              'pse_noframe': noframe_pse,
                                              'std_noframe': noframe_std})

# calculate group mean PSE and SEM for each frame angle, with no frame, and mean over all frame angles
group_pse = {}
group_sem = {}
group_offset = {'frame': {}, 'noframe': {}}
group_std = {}
group_std_sem = {}
for cond in conditions:
    group_pse[cond] = np.mean(pse[cond], axis=0)
    group_sem[cond] = (np.std(pse[cond], axis=0))/np.sqrt(len(subjects))
    group_std[cond] = np.nanmean(std_frame[cond], axis=0)
    group_std_sem[cond] = (np.nanstd(std_frame[cond], axis=0))/np.sqrt(len(subjects))
    means_over_frames = [mean_frames_pse[s][cond] for s in subjects]
    group_offset['frame'][cond] = {'mean': np.mean(means_over_frames),
                'sem': np.std(means_over_frames)/len(subjects)}
    means_noframe = [noframe_pse[cond][s] for s in subjects]
    group_offset['noframe'][cond] = {'mean': np.mean(means_noframe),
                'sem': np.std(means_noframe)/len(subjects)}


# plot mean PSE over all subjects
fig2 = plt.figure()
ax2 = fig2.add_subplot(121)
cdict = {'CCW': 'blue', 'NONE': 'green', 'CW': 'red'}
for i_cond, cond in enumerate(conditions):
    ax2.fill_between(frames, (group_pse[cond]-group_sem[cond]),
                     (group_pse[cond]+group_sem[cond]), color=cdict[cond],
                     interpolate=True, lw=0, alpha=0.2)
    ax2.plot(frames, group_pse[cond], 'o-', color=cdict[cond], markersize=7,
             linewidth=2.5)
plt.ylim((-10,6))
#    plt.title('Mean PSE and SEM over subjects')
plt.ylabel('PSE (deg)')
plt.xlabel('frame angle (deg)')
plt.legend(conditions, loc=4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)  
ax2.tick_params(axis='both', which='both', top='off', right='off')

# plot mean PSE over subjects, averaged over frame angles vs. without a frame
ax3 = fig2.add_subplot(122)
for i_cond, cond in enumerate(conditions):
    ax3.errorbar(0, group_offset['frame'][cond]['mean'],
                 yerr=group_offset['frame'][cond]['sem'], ecolor=cdict[cond])
    plt.plot(0, group_offset['frame'][cond]['mean'], 'o', markersize=10,
             color=cdict[cond])
    
    ax3.errorbar(1, group_offset['noframe'][cond]['mean'],
                 yerr=group_offset['noframe'][cond]['sem'], ecolor=cdict[cond])
    plt.plot(1, group_offset['noframe'][cond]['mean'], 'o', markersize=10,
             color=cdict[cond])

plt.xticks([0, 1], ['mean over frames', 'no frame'])
plt.ylim((-4,2))
plt.xlim((-1,2))
plt.ylabel('PSE (deg)')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)  
ax3.tick_params(axis='both', which='both', top='off', right='off')
plt.tight_layout() 

# plot mean SD over all subjects
fig3 = plt.figure()
ax4 = fig3.add_subplot(111)
for i_cond, cond in enumerate(conditions):
    ax4.fill_between(frames, (group_std[cond]-group_std_sem[cond]),
                     (group_std[cond]+group_std_sem[cond]), color=cdict[cond],
                     interpolate=True, lw=0, alpha=0.2)
    ax4.plot(frames, group_std[cond], 'o-', color=cdict[cond], markersize=7,
             linewidth=2.5)

plt.ylabel('SD (deg)')
plt.xlabel('frame angle (deg)')
plt.legend(conditions, loc=4)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)  
ax4.tick_params(axis='both', which='both', top='off', right='off')

plt.show()
