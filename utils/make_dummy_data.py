from pathlib import Path 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


def main():
    # get relative path to data sources (i.e. ./../data/sources)
    path_to_components = Path(__file__).parent.parent / 'data' / 'components.csv'

    # read the data
    data = pd.read_csv(path_to_components, index_col=0)

    # now we can randomly scale each component and add them to make a synthetic dataset
    # say we can make 384 samples 
    df = np.zeros((data.shape[0], 384))
    cols = []
    for nth_sample in range(384):
        
        # idk just picking an arbitrary low and high
        scaling_factors = np.random.uniform(low=50,high=500,size=(4, 1))
        rescaled_data = (data.values @ scaling_factors).flatten()

        # add some noise to the data maybe like 20% the low value
        noise_power = 1.2 * 500
        noise = np.random.randn(len(rescaled_data)) * noise_power

        # add noise and append sample name
        df[:,nth_sample] = rescaled_data + noise
        cols.append(f'Sample {nth_sample+1}')

    # cast the array to a df
    df = pd.DataFrame(df)
    df.index = data.index
    df.index.name = 'Wavel.'
    df.columns = cols

    # export data to csv
    df.to_csv(path_to_components.parent / 'test_data.csv', index=True)
   
if __name__ == '__main__':

    print(f'Running file: {Path(__file__).name}.') 
    main()   