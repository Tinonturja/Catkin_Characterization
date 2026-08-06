import os
import pandas as pd
class Data:
    def __init__(self, data_path:str):
        self.data_path = data_path
        self.df = None
        self.load_data()

    def load_data(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_excel(self.data_path, sheet_name='catkin')
        else:
            raise FileNotFoundError(f"The file {self.data_path} does not exist.")

data_set = Data(data_path ='./data of biofilm.xlsx')
data_df = data_set.df
print(data_df.head())