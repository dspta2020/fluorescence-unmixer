from pathlib import Path 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


def check_data_wavels(source_dfs):
    """TODO this could be eliminated with default interpolation behavior"""

    # i think all the files should follow the same wavelength range and step 
    wavel_start = [df.index.values.min() for df in source_dfs.values()]
    wavel_end = [df.index.values.max() for df in source_dfs.values()]
    wavel_step = [np.unique(np.diff(df.index.values))[0] for df in source_dfs.values()]    

    # check all the dfs
    check_flag = np.all(np.all(wavel_start == wavel_start[0]) and np.all(wavel_end == wavel_end[0]) and np.all(wavel_step == wavel_step[0]))

    # either return or raise an exception
    if check_flag:
        return
    else:
        raise Exception('Error: Inconsistent wavelength sampling.')
    
def plot_df_out(df_out):

    x = df_out.index.values
    y = df_out.values / df_out.values.max()

    plt.figure()
    plt.plot(x, y, linewidth=2)
    plt.legend(df_out.columns)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Emission')
    plt.show()

def main():

    # get relative path to data sources (i.e. ./../data/sources)
    path_to_sources = Path(__file__).parent.parent / 'data' / 'sources'

    # get the csv files and read in the data
    source_dict = {file.stem:pd.read_csv(file, index_col=0) for file in path_to_sources.glob('*.csv')}
    
    # check to make sure the wavelength sampling is consistent before combining data
    check_data_wavels(source_dfs=source_dict)

    # make a new df to write to csv
    source_dfs = [df for df in source_dict.values()]
    df_out = pd.DataFrame()
    df_out.index = source_dfs[0].index # set the outfile index to one of the input indices
    df_out.index.name = 'Wavel.' # to match TECAN export csvs

    # add the columns to the output df
    for dye, df in source_dict.items():
        df_out.loc[:,dye] = df.loc[:,'Emission']

    #plot if you want 
    # plot_df_out(df_out=df_out)
    
    # save data
    outfile_path = path_to_sources.parent
    df_out.to_csv(outfile_path / 'components.csv', index=True)


if __name__ == '__main__':

    print(f'Running file: {Path(__file__).name}.') 
    main()