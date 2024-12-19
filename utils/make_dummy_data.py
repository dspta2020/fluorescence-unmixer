from pathlib import Path 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


def main():
    
    # get relative path to data sources (i.e. ./../data/sources)
    path_to_components = Path(__file__).parent.parent / 'data' / 'components.csv'


if __name__ == '__main__':

    print(f'Running file: {Path(__file__).name}.') 
    main()
