
# ####### General Purpose Tools ####### #
#   
from genericpath import exists
from os import path, makedirs                                     #
import numpy as np
import pandas as pd
from scipy import stats
from datetime import date 
import sys 


# #### general utilities #### #
def generate_RGB(number):
    return [((1/255)*np.random.randint(0, 255), (1/255)*np.random.randint(0, 255), 
                (1/255)*np.random.randint(0, 255)) for num in range(number)]

def save_to_hdf(data_frame, save_path, file_name, key):
    if not path.exists(save_path):
        makedirs(save_path)
    data_frame.to_hdf(path.join(save_path, file_name), key = key)

# #######    useful functions    ####### #
#       Pearson Correlation              #
# useful for portfolio construction      #
# ###################################### #

def pearson_pandas(frame):
    return frame.corr(method = 'pearson').iloc[0,1]

def pearson_scipy(frame):
    return stats.pearsonr(frame.iloc[:,0], frame.iloc[:, 1])[0]

def pearson_correlation(frame, mode = 'pandas'):
    """
    computes the covariance matrix and returns 01 element
    """
    if len(frame.columns) != 2:
        print('calculation of pearson correlation needs a dataframe of two columns')
        return 
    else:
        pearson = {'pandas': pearson_pandas, 'scipy': pearson_scipy}[mode](frame)
        return pearson






