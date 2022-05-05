#import tensorflow as tf
import pandas as pd
import numpy as np
class NetModel:
    def __init__(self):
        self.ok=1
        self.fname=''
    def load_data(self):
        data = pd.read_excel(self.fname) #reading file
        #print(data)
        height,width = data.shape
        print(height,width,type(data))

        x = np.zeros((height,width),dtype=float)

        for i in range(0,height):
            for j in range(0,width):
                x[i][j] = data.iloc[i,j]
        self.xlist=x[:,0]
        self.ylist=x[:,1]
        print(self.ylist)


