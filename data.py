import os
import pandas as pd
import numpy as np
import torch
class Data:
    def __init__(self, data_path:str):
        self.data_path = data_path
        self.kinetic_df = None
        self.isotherm_df = None

        # data from the kinetics data
        self.time = None
        self.adsorption_amount = None
        self.time_tensors = None
        self.qt_tensors = None
        self.qe_tensors = None

        # synthetic data (collocation data)
        self.t_min = 0.0
        self.t_max = 180.0
        self.test_time = torch.linspace(self.t_min, self.t_max, steps=100)

        # processing data
        self.t_mean = None
        self.t_std = None
        self.qt_mean = None
        self.qt_std = None

        # Normalize data
        self.time_normalized = None
        self.qt_normalized = None
        self.allocated_time_normalized = None
        

    def load_data(self, kinetics_data_sheetname:str, isotherm_data_sheetname:str):
        if os.path.exists(self.data_path):
            self.isotherm_df = pd.read_excel(self.data_path, sheet_name=isotherm_data_sheetname)
            self.kinetic_df = pd.read_excel(self.data_path, sheet_name = kinetics_data_sheetname)
        else:
            raise FileNotFoundError(f"The file {self.data_path} does not exist.")
    
    def preprocess_and_scale(self):
        # Take the kinetics data
        self.kinetic_df = self.kinetic_df[['Time', 'Concentration_Capacity']]
        self.time = self.kinetic_df['Time']
        self.adsorption_amount = self.kinetic_df['Concentration_Capacity']
        # get the mean and the standard deviation of this data
        self.time_tensors = torch.tensor(self.time, dtype = torch.float64).view(-1,1)
        self.qt_tensors = torch.tensor(self.adsorption_amount, dtype = torch.float64).view(-1,1)
        self.test_time = torch.tensor(self.test_time, dtype = torch.float64).view(-1,1)
        self.t_mean = self.time_tensors.mean()
        self.t_std = self.time_tensors.std()
        self.qt_mean = self.qt_tensors.mean()
        self.qt_std = self.qt_tensors.std()
        # Normalize the data
        self.time_normalized = (self.time_tensors - self.t_mean) / self.t_std 
        self.qt_normalized = (self.qt_tensors - self.qt_mean) / self.qt_std
        self.allocated_time_normalized = (self.test_time - self.t_mean) / self.t_std
        


data_set = Data(data_path ='./data of biofilm.xlsx')
data_df = data_set.df
print(data_df.head())