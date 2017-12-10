"""
Read out raw data files per subject, save as pickle and csv
"""

import pandas as pd


def read_txt(textfiles):
    # Read out data files, concatenate and store in dataframe
    datalist = []
    for textfile in textfiles:
        d = (pd.read_csv(textfile, sep=', '))
        datalist.append(d)
    df = pd.concat(datalist)
    # strip whitespace from column headers
    df.rename(columns= lambda x: x.strip(), inplace=True)
    return df

    
if __name__ == '__main__':
    import glob
    
    subjects = ['01', '03', '04', '05', '06', '07', '08', '09', '11', '13',
                '14', '15', '16', '17', '18', '19', '20']
    dlist = []
    
    for sj_ind, sj in enumerate(subjects):
        datapath = 'U:\Documents\Optokinetic experiment\data\{0}\{0}_OK_*.txt'.format(sj)
        textfiles = glob.glob(datapath)
        d = read_txt(textfiles)
        # add column with subject identifier
        d['sj_id'] = [sj_ind] * len(d.index)
        dlist.append(d)
        
    df = pd.concat(dlist)
    df.to_pickle('optokinetic_data_all_sj.pkl')
    df.to_csv('optokinetic_data_all_sj.csv')
