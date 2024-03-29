# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:08:27 2019

@author: subha
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


def backward_elimination(data, target,alpha=0.05):
    dependent_var = data[target]
    independent_vars = data.drop(target,axis=1)
    i=1
    while(True):
        independent_vars = sm.add_constant(independent_vars)
        est = sm.OLS(dependent_var, independent_vars)
        est2 = est.fit()
        p_values = est2.summary2().tables[1]['P>|t|']
        possible_eliminator = np.argmax(p_values[1:])
        if p_values[possible_eliminator]>alpha:
            independent_vars.drop(possible_eliminator,axis=1,inplace=True)
            iteration_details = "Iteration Number:"+repr(i)
            eliminator_details = "\tVariable eliminated:"+ repr(possible_eliminator)
            print(iteration_details)
            print(eliminator_details)
            i=i+1
        elif p_values[possible_eliminator]<alpha:
            return est2
        if (independent_vars.empty):
            return "No Variable is significant"
    

"""
Sample code to use this function

import sys
sys.path.append(path) #replace the path with the path where you save Backward_Elimination.py file.
import Backward_Elimination.py
data = pd.read_csv(file_name) 
backward_elimination(data,target_column_name,alpha) # data is the dataframe which includes dependent variable and all independent variables.

"""