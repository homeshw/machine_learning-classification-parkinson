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
import matplotlib.pyplot as plt_c
import os
import errno
import time
import seaborn as sns

### read data from CSV ###
def readData(file_location,headers):
    print("### Reading Data >> Start ###")
    data = pd.read_csv(file_location,sep=",",names=headers)
    print("### Reading Data >> End ###")
    return data

### standardise and split training and test sets ###
def preprocessData(data,x_columns,y_column,x_y_split):
    print("### Preprocessing Data >> Start ###")

    data = data.dropna(axis=0)

    scaler = StandardScaler() 
    x_data = scaler.fit_transform(data[x_columns])

    #x_data = data_scaled[x_columns]
    y_data = data[y_column].to_numpy()

    clensed_df_x = pd.DataFrame(x_data,columns=x_columns) 
    clensed_df_y = pd.DataFrame(y_data,columns=y_column) 

    clensed_df = pd.concat([clensed_df_x,clensed_df_y],axis=1)

    print("### Saving pre-processed data to a csv ###")
    clensed_df.to_csv('classificatiodata.csv')

    x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=x_y_split)

    print("### Preprocessing Data >> End ###")

    return x_train,x_test,y_train,y_test

class NeuralNet:

    def __init__(self,train_x,test_x,train_y,test_y,iteration_count,number_of_epoch,input_dimention):
        self.trainX = train_x
        self.testX = test_x
        self.trainY = train_y        
        self.testY = test_y
        self.in_dim = input_dimention
        self.iteration_count = iteration_count
        self.number_of_epoch = number_of_epoch

    ### define elements in the neural network ###
    def createNetwork(self,test_section,test_name,opt,layers,neu_size_list):

        print("### Defining the Model " + test_section + '.' + test_name)

        self.test_section = test_section
        self.test_name = test_name
        #act1 = ks.activations.sigmoid

        # Define model
        self.model = Sequential()
        for i in range(layers):
            if i == 0:
                self.model.add(Dense(neu_size_list[i], input_dim=self.in_dim, activation=ks.activations.relu))
            else:
                self.model.add(Dense(neu_size_list[i], activation=ks.activations.relu))
        self.model.add(Dense(1, activation=ks.activations.sigmoid))
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy','Precision','Recall'])
        #print(self.model.summary())        
        print("### Defining the Model >> End ###")
    
    ### execute the neural network N times and gather performance parameters ###
    def ExecuteNEvaluate(self):

        print("### Execution and Evaluation  " + self.test_section + '.' + self.test_name)
        # initializing lists to gather metrices in each loop
        train_accuracy_l = []
        validation_accuracy_l = []
        precision_l = []
        recall_l = []
        roc_curves_l = []
        elapsed_time_l = []
        fpr = []
        tpr = []

        for i in range(self.iteration_count):
            start = time.perf_counter()
            # fit the model
            history = self.model.fit(self.trainX, self.trainY, validation_data=(self.testX, self.testY), epochs=self.number_of_epoch, verbose=0)
            elapsed = time.perf_counter() - start # elapsed time for model execution
            elapsed_time_l.append(elapsed)

            train_accuracy, test_accuracy, precision, recall = self.metricsEval(history)

            train_accuracy_l.append(train_accuracy[self.number_of_epoch-1])
            validation_accuracy_l.append(test_accuracy[self.number_of_epoch-1]) # printing the final value
            precision_l.append(precision[self.number_of_epoch-1]) # printing the final value
            recall_l.append(recall[self.number_of_epoch-1])

            if i == 0:                
                self.plotConfMatrix()
                fpr,tpr = self.getROC()

        train_accuracy_mean,train_accuracy_percentile = self.meanNpercentile(train_accuracy_l)
        validation_accuracy_mean,validation_accuracy_percentile = self.meanNpercentile(validation_accuracy_l)
        precision_mean,precision_percentile = self.meanNpercentile(precision_l)
        recall_mean,recall_percentile = self.meanNpercentile(recall_l)
        elapsed_time_mean,elapsed_time_percentile = self.meanNpercentile(elapsed_time_l)

        values_l = [train_accuracy_mean,train_accuracy_percentile,
        validation_accuracy_mean,validation_accuracy_percentile,
        precision_mean,precision_percentile,recall_mean,recall_percentile,elapsed_time_mean,elapsed_time_percentile]

        column_l = ['train_accuracy_mean','train_accuracy_percentile',
        'validation_accuracy_mean','validation_accuracy_percentile',
        'precision_mean','precision_percentile','recall_mean','recall_percentile',
        'elapsed_time_mean','elapsed_time_percentile']

        df_metrices = pd.DataFrame(values_l,index=column_l,columns=[self.test_section+'.'+self.test_name])
        print("### Execution and Evaluation  >> End ###")
        return df_metrices,fpr,tpr        

    ### common function to calculate mean and 90th percentile of a list
    def meanNpercentile(self,value_list):
        numpy_array = np.array(value_list)
        mean_val = np.mean(numpy_array)
        percentile_val = np.percentile(numpy_array,90)
        return mean_val,percentile_val
       
    # plot confusion matrix
    def plotConfMatrix(self):
        title = 'confustion_matrix_' + self.test_section + '_' + self.test_name + '.png'           
        file_name = getFolderName() + title
        predY = self.model.predict_classes(self.testX)
        conf_mat = confusion_matrix(self.testY,predY)
        conf_mat_df = pd.DataFrame(conf_mat)
        sns.heatmap(conf_mat_df, annot=True,  cmap="Blues", fmt=".0f")
        plt.title(title)
        plt.tight_layout()
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.savefig(file_name)
        plt.close()

    # plot and save ROC. each test section will save to the same grid for comparison
    def getROC(self):
        predY = self.model.predict_proba(self.testX)
        fpr, tpr, _ = roc_curve(self.testY, predY.ravel())
        return pd.Series(fpr),pd.Series(tpr)

    # fetch performance metrices from the model history
    def metricsEval(self,hist):
        train_acc = hist.history['accuracy']
        test_acc = hist.history['val_accuracy']
        precision = hist.history['precision']
        recall = hist.history['recall']
        return train_acc,test_acc,precision,recall

def plotROC(fpr,tpr,test_section,test_name):
    file_name = getFolderName() + 'roc_' + test_section + '.png'
    plt.figure(test_section)

    for i in range(len(test_name)):
        fpr_dna = fpr.iloc[:,i].dropna()
        tpr_dna = tpr.iloc[:,i].dropna()
        auc_val = auc(fpr_dna, tpr_dna)
        plt.plot(fpr_dna, tpr_dna, label=test_name[i] + ' (area = {:.3f})'.format(auc_val))
    
    plt.plot([0, 1], [0, 1], 'k--')        
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(file_name)
    plt.close()

def getFolderName():
    return 'results\\'

### test optimizer ###
def test1(nn):
    ################# common parameters ##############
    test_section = '1'
    hidden_layers = 1
    learning_rate = 0.01
    neuron_sizes = [25] # neuron sizes of each layer as a list
      

    ################ adam #####################    
    test_name_1 = 'adam'   
    optimizer = ks.optimizers.Adam(lr=learning_rate) 

    nn.createNetwork(test_section,test_name_1,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = df
    concat_fpr = fpr
    concat_tpr = tpr

    ################ section 1 - sgd #####################
    test_name_2 = 'sgd'
    optimizer = ks.optimizers.SGD(lr=learning_rate)

    nn.createNetwork(test_section,test_name_2,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    plotROC(concat_fpr,concat_tpr,test_section,[test_name_1,test_name_2])

    file_name = getFolderName() + 'results metrices test '+ test_section +'.csv'
    concat_metrices.to_csv(file_name)

### test learning rate and momentum rate ###
def test2(nn):
    ################# common parameters ################
    test_section = '2'
    hidden_layers = 1
    neuron_sizes = [25] # neuron sizes of each layer as a list
    
    ################ section 2 - adam #####################
    
    test_name_1 = 'lr-0.01'
    learning_rate = 0.01    
    optimizer = ks.optimizers.Adam(lr=learning_rate)

    nn.createNetwork(test_section,test_name_1,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = df
    concat_fpr = fpr
    concat_tpr = tpr

    ################ section 2 - lr - 0.001 #####################
    test_name_2 = 'lr-0.001'
    learning_rate = 0.001
    optimizer = ks.optimizers.Adam(lr=learning_rate)

    nn.createNetwork(test_section,test_name_2,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    ################ section 2 - lr 0.1 #####################
    test_name_3 = 'lr-0.1'
    learning_rate = 0.1
    optimizer = ks.optimizers.Adam(lr=learning_rate)

    nn.createNetwork(test_section,test_name_3,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    ################ section 2 - m 0.5 #####################
    test_name_4 = 'm-0.5'
    learning_rate = 0.01
    optimizer = ks.optimizers.SGD(lr=learning_rate,momentum = 0.5)

    nn.createNetwork(test_section,test_name_4,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    ################ section 2 - m 0.75 #####################
    test_name_5 = 'm-0.75'
    learning_rate = 0.01
    optimizer = ks.optimizers.SGD(lr=learning_rate,momentum = 0.75)

    nn.createNetwork(test_section,test_name_5,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    ################ section 2 - m 0.9 #####################
    test_name_6 = 'm-0.9'
    learning_rate = 0.01
    optimizer = ks.optimizers.SGD(lr=learning_rate,momentum = 0.9)     

    nn.createNetwork(test_section,test_name_6,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    plotROC(concat_fpr,concat_tpr,test_section,[test_name_1,test_name_2,test_name_3,test_name_4,test_name_5,test_name_6])

    file_name = getFolderName() + 'results metrices test '+ test_section +'.csv'
    concat_metrices.to_csv(file_name)

### test number of layers ###
def test3(nn):
    ################ common parameters ########################
    test_section = '3'
    
    learning_rate = 0.01   
    optimizer = ks.optimizers.Adam(lr=learning_rate)
    ################ section 3 - layers-2 #####################
    
    test_name_1 = 'layers-1'    
    hidden_layers = 1
    neuron_sizes = [25] # neuron sizes of each layer as a list       

    nn.createNetwork(test_section,test_name_1,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = df
    concat_fpr = fpr
    concat_tpr = tpr

    ################ section 3 - layers-2 #####################
    
    test_name_2 = 'layers-2'    
    hidden_layers = 2
    neuron_sizes = [25,25] # neuron sizes of each layer as a list       

    nn.createNetwork(test_section,test_name_2,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    ################ section 3 - layers-3 #####################
    test_name_3 = 'layers-3'
    hidden_layers = 3
    neuron_sizes = [25,25,25] # neuron sizes of each layer as a list      

    nn.createNetwork(test_section,test_name_3,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    ################ section 3 - layers-4 #####################
    test_name_4 = 'layers-4'
    hidden_layers = 4
    neuron_sizes = [25,25,25,25] # neuron sizes of each layer as a list      

    nn.createNetwork(test_section,test_name_4,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    plotROC(concat_fpr,concat_tpr,test_section,[test_name_1,test_name_2,test_name_3,test_name_4])
 
    file_name = getFolderName() + 'results metrices test '+ test_section +'.csv'
    concat_metrices.to_csv(file_name)

### test neuron combinations ###
def test4(nn):
    ################ common parameters ########################
    test_section = '4'    
    learning_rate = 0.01    
    optimizer = ks.optimizers.Adam(lr=learning_rate)
    hidden_layers = 4
     

    ################ section 4 - neurons - 5 #####################    
    test_name_1 = 'neurons-5'
    neuron_sizes = [5,5,5,5] # neuron sizes of each layer as a list       

    nn.createNetwork(test_section,test_name_1,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = df
    concat_fpr = fpr
    concat_tpr = tpr

    ################ section 4 - neurons - 10 #####################
    test_name_2 = 'neurons-10'
    neuron_sizes = [10,10,10,10] # neuron sizes of each layer as a list

    nn.createNetwork(test_section,test_name_2,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    ################ section 4 - neurons - 15 #####################
    test_name_3 = 'neurons-15'
    neuron_sizes = [15,15,15,15] # neuron sizes of each layer as a list

    nn.createNetwork(test_section,test_name_3,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    ################ section 4 - neurons - 20 #####################
    test_name_4 = 'neurons-20'
    neuron_sizes = [20,20,20,20] # neuron sizes of each layer as a list

    nn.createNetwork(test_section,test_name_4,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    ################ section 4 - neurons - 25 #####################
    test_name_5 = 'neurons-25'
    neuron_sizes = [25,25,25,25] # neuron sizes of each layer as a list

    nn.createNetwork(test_section,test_name_5,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    ################ section 4 - neurons - 25,20,15,10 #####################
    test_name_6 = 'neurons-25,20,15,10'
    neuron_sizes = [25,20,15,10] # neuron sizes of each layer as a list

    nn.createNetwork(test_section,test_name_6,optimizer,hidden_layers,neuron_sizes)    
    df,fpr,tpr = nn.ExecuteNEvaluate()
    concat_metrices = pd.concat([concat_metrices,df],axis=1)
    concat_fpr = pd.concat([concat_fpr,fpr],axis=1)
    concat_tpr = pd.concat([concat_tpr,tpr],axis=1)

    plotROC(concat_fpr,concat_tpr,test_section,[test_name_1,test_name_2,test_name_3,test_name_4,test_name_5,test_name_6])

    file_name = getFolderName() + 'results metrices test '+ test_section +'.csv'
    concat_metrices.to_csv(file_name)
    
def main():
    print("### Main >> Start ###")

    # creating a folder to save all the results
    try:
        os.makedirs('results')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

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

    train_test_split = 0.4
       
    x_train,x_test,y_train,y_test = preprocessData(data,feature_columns,result_column,train_test_split)

    input_dimention = len(feature_columns)
    number_of_iterations = 10 # number of iterations
    number_of_epoch = 25
    # initialize the NeuralNet object with the training and test data sets
    nn = NeuralNet(x_train,x_test,y_train,y_test,number_of_iterations,number_of_epoch,input_dimention)
    
    test1(nn)
    test2(nn)
    test3(nn)
    test4(nn)
    
    print("### Main >> End ###")

if __name__ == "__main__":
    main()