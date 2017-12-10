"""
Predict performance with optokinetic stimulus and frame from the no_frame data
with only the optokinetic stimulus, and from the NONE condition with only the
frame but no optokinetic rotation.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import pandas as pd


def calc_rmse(prediction, data):
    error = prediction - data
    rmse = np.sqrt(np.mean(np.square(error)))
    return rmse


def significance_label(axis, rects, pvalues, yvalues, serrors):
    """
    Place asterisk above bar in bar graph, conditional on the p-value
    """
    for rect, p, y, se in zip(rects, pvalues, yvalues, serrors):
        if y < 0.0:
            text_y = y - se - 1.8
        else:
            text_y = y + se
            
        if p < 0.01:
            axis.text(rect.get_x() + rect.get_width()/1.8, text_y, '**', ha='center', va='bottom')
        elif p < 0.05:
            axis.text(rect.get_x() + rect.get_width()/1.8, text_y, '*', ha='center', va='bottom')
        else:
            pass


with open('pse_all_sj.pkl', 'rb') as fhandle:
    pse = pickle.load(fhandle)
with open('std_all_sj.pkl', 'rb') as fhandle:
    std_frame = pickle.load(fhandle)
with open('noframe_pse_all_sj.pkl', 'rb') as fhandle:
    noframe_pse = pickle.load(fhandle)
  
condition = ['CCW', 'CW']
colour = ['blue', 'red']
subjects = np.arange(np.shape(pse['CCW'])[0])
columns = ['subject', 'rmse', 'R2', 'B0', 'B1', 'B0_SE', 'B1_SE', 'B0_t',
           'B1_t', 'B0_p', 'B1_p', 'noframe_R2']
output = {'CCW': pd.DataFrame(index=subjects, columns=columns),
          'CW': pd.DataFrame(index=subjects, columns=columns)}

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)
axes = [ax1, ax2]
fig1.text(0.5, 0.01, 'measured PSE without optokinetic stimulation',
          ha='center')
for sj in subjects:
    for col, ax, cond in zip(colour, axes, condition):
        
        output[cond]['subject'][sj] = sj
        
        # Regressors
        modeldata = {}
        modeldata['xdata'] = pse['NONE'][sj, 0:8]
        modeldata['ydata'] = pse[cond][sj, 0:8]
        
        # Linear regression model for CCW data
        model_lr = sm.ols(formula='ydata ~ xdata', data=modeldata)
        result_lr = model_lr.fit()
        
        # Regression results and stats
        output[cond]['rmse'][sj] = calc_rmse(result_lr.predict(), modeldata['ydata'])
        output[cond]['R2'][sj] = result_lr.rsquared
        output[cond]['B0'][sj], output[cond]['B1'][sj] = result_lr.params
        output[cond]['B0_SE'][sj], output[cond]['B1_SE'][sj] = result_lr.bse
        output[cond]['B0_t'][sj], output[cond]['B1_t'][sj] = result_lr.tvalues
        output[cond]['B0_p'][sj], output[cond]['B1_p'][sj] = result_lr.pvalues
    
        # regression lines
        xline = np.linspace(-15,16)
        yline = xline* output[cond]['B1'][sj] + output[cond]['B0'][sj]
    
        # plot data and regression lines
        ax.plot([-15,15], [-15,15], '-', color='black')
        ax.plot(xline, yline, '-', color=col)
        ax.plot(modeldata['xdata'], modeldata['ydata'], '.', color=col)
        ax.set_title(cond)

output['CCW'].to_csv('regresult_CCW.txt')
output['CW'].to_csv('regresult_CW.txt')
        
ax1.set_xlim([-15, 15])
ax1.set_ylim([-25, 25])
ax1.set_ylabel('measured PSE with optokinetic stimulation')
ax2.set_xlim([-15, 15])
ax2.set_ylim([-25, 25])

# Plot regression weights for all subjects
fig2, (ax3, ax4) = plt.subplots(2,1)
width = 0.35
# Beta 1 (slope)
bars1 = ax3.bar(subjects, output['CCW']['B1'], width, color='blue',
                yerr=output['CCW']['B1_SE'], error_kw={'ecolor': 'black'})
bars2 = ax3.bar(subjects + width, output['CW']['B1'], width, color='red',
                yerr=output['CW']['B1_SE'], error_kw={'ecolor': 'black'})
ax3.set_title('slope beta weight')
ax3.set_xticks(subjects + width/2)
ax3.set_xticklabels(map(str, subjects+1))
ax3.set_xlabel('participant')
ax3.set_xlim((0-width, len(subjects)))
ax3.tick_params(axis='both', which='both', top='off', right='off')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_ylim((0, 2.5))
# Place text labels over bars to indicate significance level
significance_label(ax3, bars1, output['CCW']['B1_p'], output['CCW']['B1'], output['CCW']['B1_SE'])
significance_label(ax3, bars2, output['CW']['B1_p'], output['CW']['B1'], output['CW']['B1_SE'])

# Beta 0 (intercept)
bars3 = ax4.bar(subjects, output['CCW']['B0'], width, color='blue',
                yerr=output['CCW']['B0_SE'], error_kw={'ecolor': 'black'})
bars4 = ax4.bar(subjects + width, output['CW']['B0'], width, color='red',
                yerr=output['CW']['B0_SE'], error_kw={'ecolor': 'black'})
ax4.set_title('intercept beta weight')
ax4.set_xticks(subjects + width/2)
ax4.set_xticklabels(map(str, subjects+1))
ax4.set_xlabel('participant')
ax4.set_xlim((0-width, len(subjects)))
ax4.tick_params(axis='both', which='both', top='off', right='off')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.set_ylim((-10, 12))
# Place text labels over bars to indicate significance level
significance_label(ax4, bars3, output['CCW']['B0_p'], output['CCW']['B0'], output['CCW']['B0_SE'])
significance_label(ax4, bars4, output['CW']['B0_p'], output['CW']['B0'], output['CW']['B0_SE'])
plt.tight_layout()


# Predict SD from bias (PSE)
intercept = {}
slope = {}
corr = {}
cols = ['blue', 'green', 'red', 'black']
for icond, cond in enumerate(['CCW', 'NONE', 'CW', 'ALL']):
    # regression
    mdata = {}
    if cond == 'ALL':
        mdata['bias'] = abs(np.concatenate((pse['CCW'][:, 0:8].flatten(),
                                            pse['NONE'][:, 0:8].flatten(),
                                            pse['CW'][:, 0:8].flatten())))
        mdata['sigma'] = np.concatenate((std_frame['CCW'][:, 0:8].flatten(),
                                         std_frame['NONE'][:, 0:8].flatten(),
                                         std_frame['CW'][:, 0:8].flatten()))
    else:
        mdata['bias'] = abs(pse[cond][:, 0:8].flatten())
        mdata['sigma'] = std_frame[cond][:, 0:8].flatten()
    model_sd_pse = sm.ols(formula='sigma ~ bias', data=mdata)
    result_sd_pse = model_sd_pse.fit()
    intercept[cond], slope[cond] = result_sd_pse.params
    corr[cond] = np.sqrt(result_sd_pse.rsquared)
    
    # regression lines
    xline = np.linspace(0,20)
    yline = xline* slope[cond] + intercept[cond]
    
    # scatter plots
    if cond == 'ALL':
        ax = plt.subplot2grid((2,3), (1,0), colspan=3)
        ax.set_xlabel('bias (deg)')
    else:
        ax = plt.subplot2grid((2,3), (0,icond))
    ax.plot(mdata['bias'], mdata['sigma'], 'o', color=cols[icond])
    ax.plot(xline, yline, '-', color=cols[icond])
    ax.text(0, 15, 'R = {}'.format(round(corr[cond], 2)))
    ax.set_xlim((0, 20))
    ax.set_ylim((0, 20))
    ax.set_ylabel('SD (deg)')
    ax.set_title(cond)
    plt.tight_layout()
