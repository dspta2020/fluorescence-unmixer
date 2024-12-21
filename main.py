import os
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, send_file

import pandas as pd
import numpy as np
import scipy.optimize

# sets the working dir path to be the file directory
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
RESULTS_FOLDER = './results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


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

# this is the homepage of the application and will return the index.html when someone access this
@app.route('/')
def index():
    # this html is in the templates folder
    return render_template('index.html')

# here we want to only allow POST methods which sends data to the server (ie this app)
@app.route('/upload', methods=['POST'])
def upload_files():

    # check that files have been selected
    if 'data_file' not in request.files or 'components_file' not in request.files:
        return "Please upload both files.", 400

    # not sure what this is doing here
    data_file = request.files['data_file']
    components_file = request.files['components_file']

    # get the paths to data
    data_path = os.path.join(UPLOAD_FOLDER, 'data.csv')
    components_path = os.path.join(UPLOAD_FOLDER, 'components.csv')
    
    # save the files into the uploads folder
    data_file.save(data_path)
    components_file.save(components_path)

    # read in the data
    data = pd.read_csv(data_path, index_col=0)
    components = pd.read_csv(components_path, index_col=0)

    # get the max range of wavelengths sampled 
    wl_range = np.array([data.index.max(), data.index.min(), components.index.max(), components.index.min()])
    
    # interpolate to ensure the same wavelength sampling
    data = interpolate_df(wl_range.max(), wl_range.min(), data)
    components = interpolate_df(wl_range.max(), wl_range.min(), components)

    # loop through each test sample and fit
    for _, nth_sample in enumerate(data.columns):
        '''
        For reference the scipy documentation states the nnls algorithm is from:

            [2] : Bro, Rasmus and de Jong, Sijmen, "A Fast Non-Negativity-
            Constrained Least Squares Algorithm", Journal Of Chemometrics, 1997,
            :doi:`10.1002/(SICI)1099-128X(199709/10)11:5<393::AID-CEM483>3.0.CO;2-L`
        '''
        nth_fit_coeffs = scipy.optimize.nnls(components.values, data[nth_sample])
        nth_fit_components = components.values * nth_fit_coeffs[0]
        nth_fit_full = components.values @ nth_fit_coeffs[0]

        # use pandas to write the results to a df then export to csv
        df = pd.DataFrame()
        df['Raw Data'] = data[nth_sample]
        df['Fit Data'] = nth_fit_full
        for nth_component, col in enumerate(components.columns):
            df[f'{col} fit'] = nth_fit_components[:, nth_component]
        for nth_component, col in enumerate(components.columns):
            df[f'{col} coeff'] = np.nan
            df.loc[df.index[0], f'{col} coeff'] = nth_fit_coeffs[0][nth_component]

        # save the results path to the results folder
        result_path = os.path.join(RESULTS_FOLDER, f'{nth_sample}.csv')
        df.to_csv(result_path)

    # after all the files are saved then redirect users to the results page below
    return redirect(url_for('download_results'))


@app.route('/results')
def download_results():
    files = sorted([file for file in os.listdir(RESULTS_FOLDER)])
    return render_template('results.html', files=files)


@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(RESULTS_FOLDER, filename)
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
