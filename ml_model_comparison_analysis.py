# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
# import models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def load_models(name:str,):
    model_dic = {
        'Polynomial_Regressor': lambda: Pipeline('poly')
    }