import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread


def get_average_df(pattern, runs_dir):
    run_names = os.listdir(runs_dir)
    run_names = filter(lambda s: pattern in s, run_names)
    df_list = [pd.read_csv(os.path.join(runs_dir, name)) for name in run_names]
    df = pd.


get_average_df("weight0p0000", "pseudo_labels_runs")
