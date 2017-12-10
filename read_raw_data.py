"""
Read out raw data files per subject, save as pickle and csv
"""

import pandas as pd

def read_txt(textfiles):
    # Read out data files, concatenate and store in dataframe
    datalist    = []
    for textfile in textfiles:
        d = (pd.read_csv(textfile, sep=', '))
        datalist.append(d)
    df = pd.concat(datalist)
    df.rename(columns= lambda x: x.strip(), inplace=True)            # strip whitespace from column headers
    return df

    
if __name__ == '__main__':
    import glob
    
    subjects    = ['01', '03', '04', '05', '06', '07', '08', '09', '11', '13', '14', '15', '16', '17', '18', '19', '20']
    dlist = []
    
    for sj_ind, sj in enumerate(subjects):
        datapath    = 'U:\Documents\Optokinetic experiment\data\{0}\{0}_OK_*.txt'.format(sj)
        textfiles   = glob.glob(datapath)
        d = read_txt(textfiles)
        d['sj_id'] = [sj_ind] * len(d.index)              # add column with subject identifier
        dlist.append(d)
        
    df = pd.concat(dlist)
    df.to_pickle('optokinetic_data_all_sj.pkl')         # save dataframe to pickle file
    df.to_csv('optokinetic_data_all_sj.csv')
