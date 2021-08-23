import pandas as pd
import keras as ks
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import errno

def readData(file_location,headers):
    print("### Reading Data >> Start ###")
    data = pd.read_csv(file_location,sep=",",names=headers)
    print("### Reading Data >> End ###")
    return data

def main():
    print("### Main >> Start ###")

    file_location = 'train_data.txt'
    feature_columns = ['Jitter_local','Jitter_local_absolute','Jitter_rap','Jitter_ppq5','Jitter_ddp',
    'Shimmer_local','Shimmer_local_dB','Shimmer_apq3','Shimmer_apq5','Shimmer_apq11','Shimmer_dda',
    'AC','NTH','HTN',
    'Median_pitch','Mean_pitch','Standard_deviation','Minimum_pitch','Maximum_pitch',
    'Number_of_ pulses','Number_of_periods','Mean_period','Standard_deviation_of_period',
    'Fraction_of_locally_unvoiced_frames','Number_of_voice_breaks','Degree_of_voice_breaks']
    dropped_columns = ['UPDRS']
    result_column = ['class']
    
    column_names = feature_columns + dropped_columns + result_column
    
    data = readData(file_location,column_names)

    print("class distribution:")
    print(data.groupby(['class']).size().reset_index(name='counts'))

    try:
        os.makedirs('visualizations')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    #plot histograms of each attributes

    for i in range(len(feature_columns)):
        n, bins, patches = plt.hist(x=data.iloc[:,i], bins='auto', color='#607c8e',alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(feature_columns[i])
        #plt.text(23, 45, r'$\mu=15, b=3$')
        plt.savefig('visualizations' + '\\' + feature_columns[i])
        plt.close()
    

if __name__ == "__main__":
    main()