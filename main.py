import time
from pathlib import Path

import pandas as pd 
import numpy as np 
import scipy.optimize
import matplotlib.pyplot as plt


def interpolate_df(range_max, range_min, df):

    # number of samples to process 
    num_samples = df.shape[1]
    # the new number of wavelengths sampled 
    new_len = range_max - range_min

    # preallocate new array
    new_array = np.zeros((new_len+1, num_samples))
    
    # setup interpolation
    x = df.index.values
    xnew = np.arange(range_min, range_max+1)

    # loop to interpolate each sample
    for nth_sample in range(num_samples):
        """
        For reference numpys interp documetation:
            One-dimensional linear interpolation for monotonically increasing sample points.

            Returns the one-dimensional piecewise linear interpolant to a function
            with given discrete data points (`xp`, `fp`), evaluated at `x`.
        """
        y = df.iloc[:,nth_sample]
        ynew = np.interp(xnew,x,y)

        # add to the new array
        new_array[:,nth_sample] = ynew
    
    # format output df
    new_df = pd.DataFrame(new_array)
    new_df.index = df.index
    new_df.columns = df.columns

    return new_df

def main():

    # get the paths to data
    path_to_data = Path('./data/test_data.csv')
    path_to_components = Path('./data/components.csv')

    # read in the data
    data = pd.read_csv(path_to_data, index_col=0)
    components = pd.read_csv(path_to_components, index_col=0)

    # determine the max wavelength range
    wl_range = np.array([data.index.max(), data.index.min(), components.index.max(), components.index.min()])
    
    # interpolate the data so its on same wavelength interval 
    data = interpolate_df(wl_range.max(),wl_range.min(),data)
    components = interpolate_df(wl_range.max(),wl_range.min(),components)
    
    # loop through each test sample and fit
    for n, nth_sample in enumerate(data.columns):
        '''
        For reference the scipy documentation states the nnls algorithm is from:

            [2] : Bro, Rasmus and de Jong, Sijmen, "A Fast Non-Negativity-
            Constrained Least Squares Algorithm", Journal Of Chemometrics, 1997,
            :doi:`10.1002/(SICI)1099-128X(199709/10)11:5<393::AID-CEM483>3.0.CO;2-L`
        '''
        nth_fit_coeffs = scipy.optimize.nnls(components.values, data[nth_sample])
        nth_fit_components = components.values * nth_fit_coeffs[0]
        nth_fit_full = components.values @ nth_fit_coeffs[0]


if __name__ == '__main__':

    print(f"Running File: {Path(__file__).name}")

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
