import pandas as pd
import numpy as np

class preprocess():

    def setup(self,file,*args):
        """
        This function takes in the file wich is needed and returns a matrix.
        Paramters:
            Input:
                file: Csv file that should be used
                *args: the args for removal
            Output:
                numpy.array
        """
        dataframe=pd.read_csv(file)
        for n in args:
            del dataframe[n]
        return dataframe.values

    def combine_arrays(self,a,b):
        """
        Combines two arrays (a,b) with np.vstack((a,b))
        """
        return np.vstack((a,b))

    def word_setup(self,file,*args):
        """
        Takes in a file and *args as input and outputs a csv called new_csv.csv.
        """
        dataframe=pd.read_csv(file)
        for n in args:
            del dataframe[n]
        dataframe.to_csv("new_csv.csv",header=False,index=False)
