import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import numpy.random as npr
import random
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import sys

sys.setrecursionlimit(1000)

class Regression:
    def __init__(self,csv_file=None,data=None,values=None):
        if(data is None and csv_file is not None):
            pass