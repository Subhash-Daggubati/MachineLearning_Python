# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:05:01 2019

@author: subha
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from os import path 
import scipy.stats as stats


class Data:
    def __init__(self, train_features_file, train_target_file, test_file, key_col, target_col,report=None):
        self.key_col = key_col
        self.report = report
        self.report_flag = not (self.report is None)
        if self.report_flag:
            self.report.start_report()
        self.train_df = self._create_train_df(train_features_file,train_target_file)
        self.test_df = self._create_test_df(test_file)
        self.target_col = [target_col]
        self.num_features_list = self._create_num_features_list(self.test_df)
        self.cat_features_list = self._create_cat_features_list(self.test_df)
        
        
    def _create_cat_features_list(self,data):
        info = data.dtypes
        cat_features_list = list(info[info=='object'].index.values) 
        return cat_features_list
    
    def _create_num_features_list(self,data):
        info = data.dtypes
        return list(info[info!='object'].index.values)
    
    def get_cat_features_list(self):
        return self.cat_features_list
    
    def get_num_features_list(self):
        return self.num_features_list
    
    def add_df_head_to_report(self,dataframe_name,df):
        self.report.add_header(dataframe_name)
        self.report.add_text(f"First 5 rows of the {dataframe_name} are displayed below using df.head(5)")
        self.report.add_df_as_table(df.head())
    
    def _create_train_df(self,train_features_file, train_target_file):
        train_features = self._load_data(train_features_file)
        train_target = self._load_data(train_target_file)
        train_df= train_features.merge(train_target,how ='inner')
        if self.report_flag:
            self.add_df_head_to_report("Training Features Dataframe",train_features)
            self.add_df_head_to_report("Training Target Dataframe",train_target)
            self.add_df_head_to_report("Consolidated Training Dataframe",train_df)
#        train_df = self.clean_missing_data(train_df)
#        train_df = self.clean_duplicate_data(train_df)
        return train_df
    
    def _create_test_df(self,test_file):
        test_df = self._load_data(test_file)
        if self.report_flag:
            self.add_df_head_to_report("Test",test_df)
        return test_df
    
    def _load_data(self,file):
        return pd.read_csv(file)
    
    def clean_missing_data(self,data):
        pass
    
    def clean_duplicate_data(self,data):
        pass
        
        
class HTML_Report:
    
    def __init__(self,filename,file_path=None):
        self.filename = filename
        self.file_path = file_path
        self.report_content = "None"
        self.set_report_dir()
        
    def set_report_dir(self):
        if (self.file_path is None):
            self.file_path = os.getcwd()
        if not path.isdir(self.file_path):
            os.makedirs(self.file_path)
        self.file = path.join(self.file_path,self.filename)
        self.images = path.join(self.file_path,"Images")
        if not path.isdir(self.images):
            os.mkdir(self.images)

    def start_report(self):
        self.report_content = '''<html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>body{ margin:0 100; background:whitesmoke; }</style>
                <title> EDA </title>
            </head>
            <body>'''
    
    def add_header(self, header_text, level=2):
        if level not in range(1,7):
            raise ValueError("Invalid header level. Choose a value between 1 and 6 inclusively")
        else:
            htag = "h"+ str(level)
            html_header = "<"+htag+">" + header_text + "</"+ htag + ">"
            self.report_content = self.report_content + html_header
    
    def add_image(self,image_url):
        image = '''<img src=" ''' + image_url + '''" >'''
#        image = '''<iframe width="1000" height="1000" frameborder="0" seamless="seamless" scrolling="no" src="''' + image_url + '''.embed?width=1000&height=1000"></iframe>'''
        self.report_content = self.report_content + image
    
    def add_df_as_table(self, df):
        table = df.to_html().replace('<table border="1" class="dataframe">','<table class="table table-striped">')
        self.report_content = self.report_content + table
        
    def add_text(self,text):
        paragraph = "<p>" + text +"</p>"
        self.report_content = self.report_content + paragraph
        
    def generate_report(self):
        closing_tags = '''</body> </html>'''
        self.report_content = self.report_content + closing_tags
        f = open(self.file+".html",'w')
        f.write(self.report_content)
        f.close()
                           
class EDA_plots:
    def __init__(self,data):
        self.data = data
        self.train_df = data.train_df
        self.cat_features_list = data.cat_features_list
        self.num_features_list = data.num_features_list
        self.target_col = data.target_col
        self.image_path = data.report.images
        
    def generate_image_filename(self,col_name):
        image_file = path.join(self.image_path,col_name+".png")
        return image_file
    
    def add_variable_plot_to_report(self, col_name,file_url):
        self.data.report.add_header(col_name)
#        img_url = "/Images/" + col_name+".png"
        self.data.report.add_image(file_url)

    def plot_target_var(self):
        fig1,(ax,ax1) = plt.subplots(1,2)
        sns.distplot(self.train_df[self.target_col], bins=20, ax=ax)
        g1= sns.boxplot(self.train_df[self.target_col],ax=ax1)
        g1.set_xticklabels(g1.get_xticklabels(),rotation =30)
        file_url = self.generate_image_filename(self.target_col[0])
        plt.savefig(file_url)
        target_details = self.train_df[self.target_col].describe()
        if self.data.report_flag:
            self.add_variable_plot_to_report("Target Variable: "+self.target_col[0],file_url)
            self.data.report.add_df_as_table(target_details)
        
    def plots(self,column):
        if self.data.train_df[column].dtype.name == 'int64':
            fig2, ax2 = plt.subplots()
            sns.jointplot(column,self.target_col[0], data = self.data.train_df, kind= 'reg').annotate(stats.pearsonr)
            file_url = self.generate_image_filename(column)
            plt.savefig(file_url)
            if self.data.report_flag:
                self.add_variable_plot_to_report(column,file_url)
        else:
            value_counts = self.data.train_df[column].value_counts()
            if len(value_counts)>15:
                if self.data.report_flag:
                    self.data.report.add_text(column +" variable has more than 15 categories. Can not plot the figure.")
                return f"Can not plot the figure. {column} has more than 15 columns"
            fig3,(ax3,ax4)=plt.subplots(1,2,figsize=(15,7))
            g=sns.boxplot(column, self.target_col[0], data = self.data.train_df,ax=ax3)
            g.set_xticklabels(g.get_xticklabels(),rotation =30)
            no = np.arange(len(value_counts))
            ax4.bar( no, value_counts)
            plt.xlabel(column, fontsize=10)
            ax4.set_ylabel('value_counts', fontsize=10)
            ax3.set_ylabel(self.target_col[0], fontsize=10)
            plt.xticks(no, value_counts.index , fontsize=10, rotation=45)
            plt.title('Frequency')
            file_url = self.generate_image_filename(column)
            plt.savefig(file_url)
            if self.data.report_flag:
                self.add_variable_plot_to_report(column,file_url)
        
        

#train_features = pd.read_csv("D:\\Fall 2019\\DSDJ\\Portfolio\\DSDJ-data-SalaryPredictions\\data\\train_features.csv",header=0)
#train_salaries = pd.read_csv("D:\\Fall 2019\\DSDJ\\Portfolio\\DSDJ-data-SalaryPredictions\\data\\train_salaries.csv",header=0)
if __name__ == '__main__':
    import sys
    args_count = len(sys.argv)
    if args_count>8:
        raise ValueError("Invalid arugument length. maximum of 7 arguments are allowed")
    elif args_count >=7:
        report_name = sys.argv[6]
        train_features = sys.argv[1]
        train_targets = sys.argv[2]
        test_features = sys.argv[3]
        key_col = sys.argv[4]
        target_col = sys.argv[5]
        report_name = sys.argv[6]
        report_path = None
        if args_count ==8 : 
            report_path = sys.argv[7]
#        data = Data(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],report)
    else:
        raise ValueError("Invalid arugument length. Minimum of 6 arguments are required")
    report = HTML_Report(report_name,report_path)
    data = Data(train_features,train_targets,test_features,key_col,target_col,report)
    eda_plots = EDA_plots(data)

    eda_plots.plot_target_var()
    if data.report_flag :
        data.report.add_header("Plots of Training Features")
    
    all_features = data.num_features_list + data.cat_features_list
    all_features.remove(data.key_col)
    
    for feature in all_features:
        eda_plots.plots(feature)
        
    data.report.generate_report()

