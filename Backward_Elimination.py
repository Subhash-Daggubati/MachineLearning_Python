# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:08:27 2019

@author: subha
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

file_path = 'D:\\Fall 2019\\ML\\bodyfat.txt'
data = pd.read_csv(file_path, sep='\t',header=0)

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

output = backward_elimination(data,'Pct.BF')
output.summary()
