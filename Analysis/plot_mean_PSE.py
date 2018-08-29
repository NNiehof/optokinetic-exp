"""
Plot Figure 3 for in optokinetic paper
"""

import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import pickle

debug = False

# read in dataframe
df = pd.read_pickle('U:\Documents\Optokinetic experiment\optokinetic_data_all_sj.pkl')
subjects = sorted(df['sj_id'].unique())
conditions = ['CW', 'NONE', 'CCW']
frame_angles = ['-45', '-33.75', '-22.5', '-11.25', '0', '11.25', '22.5', '33.75', 'noframe']
frames = [-45, -33.75, -22.5, -11.25, 0, 11.25, 22.5, 33.75, 45]

# read in PSE and SD
with open('pse_all_sj.pkl', 'rb') as f:
    pse = pickle.load(f)
with open('std_all_sj.pkl', 'rb') as f:
    std_frame = pickle.load(f)
with open('noframe_pse_all_sj.pkl', 'rb') as f:
    noframe_pse = pickle.load(f)
with open('noframe_std_all_sj.pkl', 'rb') as f:
    noframe_std = pickle.load(f)

# calculate group mean PSE and SEM for each frame angle, with no frame, and mean over all frame angles
group_pse = {}
group_sem = {}
group_offset = {'frame': {}, 'noframe': {}}
group_var = {'frame': {}, 'noframe': {}}
group_std = {}
group_std_sem = {}

for cond in conditions:
    group_pse[cond] = np.mean(pse[cond], axis=0)
    group_sem[cond] = (np.std(pse[cond], axis=0))/np.sqrt(len(subjects))
    group_std[cond] = np.nanmean(std_frame[cond], axis=0)
    group_std_sem[cond] = (np.nanstd(std_frame[cond], axis=0))/np.sqrt(len(subjects))
    means_noframe = [noframe_pse[cond][s] for s in subjects]
    std_noframe = [noframe_std[cond][s] for s in subjects]
    group_offset['noframe'][cond] = {'mean': np.mean(means_noframe),
                                     'sem': np.std(means_noframe)/len(subjects)}
    group_var['noframe'][cond] = {'mean': np.mean(std_noframe),
                                  'sem': np.std(std_noframe) / len(subjects)}


# plot mean PSE over all subjects
fig = plt.figure()
ax2 = fig.add_subplot(121)
cdict = {'CCW': 'turquoise', 'NONE': 'navy', 'CW': 'crimson'}
for i_cond, cond in enumerate(conditions):
    ax2.fill_between(frames, (group_pse[cond]-group_sem[cond]),
                     (group_pse[cond]+group_sem[cond]), color=cdict[cond],
                     interpolate=True, lw=0, alpha=0.2)
    ax2.plot(frames, group_pse[cond], 'o-', color=cdict[cond], markersize=7,
             linewidth=2.5)
plt.ylim((-6, 3))
#    plt.title('Mean PSE and SEM over subjects')
plt.ylabel('PSE (deg)')
plt.xlabel('frame angle (deg)')
plt.legend(conditions, loc=4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='both', which='both', top='off', right='off')

# change the tick label for no frame
ax2.set_xlim([-60, 70])
ax2.set_xticks([-40, -20, 0, 20, 40, 60])
tick_labels = ax2.get_xticks().tolist()
tick_labels[5] = "no\nframe"
ax2.set_xticklabels(tick_labels)

# plot mean PSE without a frame
for i_cond, cond in enumerate(conditions):
    ax2.errorbar(60, group_offset['noframe'][cond]['mean'],
                 yerr=group_offset['noframe'][cond]['sem'], ecolor=cdict[cond])
    plt.plot(60, group_offset['noframe'][cond]['mean'], 'o', markersize=7,
             color=cdict[cond])

# plot mean SD over all subjects
ax4 = fig.add_subplot(122)
for i_cond, cond in enumerate(conditions):
    ax4.fill_between(frames, (group_std[cond]-group_std_sem[cond]),
                     (group_std[cond]+group_std_sem[cond]), color=cdict[cond],
                     interpolate=True, lw=0, alpha=0.2)
    ax4.plot(frames, group_std[cond], 'o-', color=cdict[cond], markersize=7,
             linewidth=2.5)

# plot mean SD without a frame
for i_cond, cond in enumerate(conditions):
    ax4.errorbar(60, group_var['noframe'][cond]['mean'],
                 yerr=group_var['noframe'][cond]['sem'], ecolor=cdict[cond])
    plt.plot(60, group_var['noframe'][cond]['mean'], 'o', markersize=7,
             color=cdict[cond])

plt.ylim((0, 6))
plt.ylabel('SD (deg)')
plt.xlabel('frame angle (deg)')
plt.legend(conditions, loc=4)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.tick_params(axis='both', which='both', top='off', right='off')

# change the tick label for no frame
ax4.set_xlim([-60, 70])
ax4.set_xticks([-40, -20, 0, 20, 40, 60])
tick_labels = ax4.get_xticks().tolist()
tick_labels[5] = "no\nframe"
ax4.set_xticklabels(tick_labels)

plt.show()
