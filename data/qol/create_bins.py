# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:35:34 2020

@author: nandakishore.m
"""
import pandas as pd
import numpy as np

def create_bin(g_vec, bins):
    if(bins == 3):
        labels = ["low","mid","high"]
    elif(bins == 5):
        labels = ["v.low","low","mid","high","v.high"]
    g_vec_bins= pd.cut(np.array(g_vec),bins,labels=labels)
    return g_vec_bins

def write_to_csv(csv,input_cols,output_cols,bins):
    for i in range(len(input_cols)):
        input_vec = csv[input_cols[i]]
        input_vec.fillna( 0, inplace=True)
        output_vec_bin = create_bin(input_vec, bins[i])
        output_bin=list(output_vec_bin)
        csv[output_cols[i]] = output_bin
    return csv

def main():  
    input_cols = ["Drought_Episodes","livestock","agriculture_land","UN_Population_Density_2015"]
    output_cols = ["drought","livestock_bin","agriculture_land_bin","pop_density"]
    bins = [3,3,3,5]
    orig_csv = pd.read_csv('20-08-2019-clusters.csv')
    csv_bin = write_to_csv(orig_csv,input_cols,output_cols,bins)
    csv_bin.to_csv("sentinel.csv")
    
    
if __name__ == "__main__":
    main()