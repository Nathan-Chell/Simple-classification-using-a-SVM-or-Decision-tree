#Perform linear classification on a dataset using a Kernal SVM

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC

import os


def get_dataset():
    
    current_path = os.getcwd()
    parrent_dir = os.path.dirname(current_path)
    data_dir = os.path.join(parrent_dir, 'data')
    
    return pd.read_csv(os.path.join(data_dir, 'Social_Network_Ads.csv'))

def plot_data_parameters(data):
    
    print("Shape: {}\nHead of data: {}".format(data.shape, data[:10]))
    
def plot_data(data, featureX, featureY):
    
    plt.plot(data[featureX], data[featureY], 'b.')
    plt.xlabel(featureX)
    plt.ylabel(featureY)
    
    plt.show()
    
def Split_data(data):
    
    X = data.iloc[:, [1,2,3]].values
    y = data.iloc[:, 4].values
    
    #Replace categorical data with numerical data
    df_X = pd.DataFrame(X, columns=['Gender', 'Age', 'EstimatedSalary'])
    df_X['Gender'].replace(['Male','Female'], [0, 1], inplace = True)
    
    df_Y = pd.DataFrame(y, columns=['Purchased'])
    
    #Normilize data
    #We want to scale the salary only
    min_max_scaler = preprocessing.MinMaxScaler()
    df_X['EstimatedSalary'] = min_max_scaler.fit_transform(df_X['EstimatedSalary'].values.reshape(-1,1))
    
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size = 0.25, random_state = 0)
    
    return X_train, X_test, y_train, y_test

def train_svm(X_train, y_train, X_test, y_test, kernels):
    
    for kernel in kernels:
        model = SVC(kernel = kernel)
        model.fit(X_train, y_train.values.ravel())
        
        print("Model score: {}, using kernal: {}, with an accuracy of: {}".format(model.score(X_train, y_train), kernel, model.score(X_test, y_test.values.ravel())))


def main():
    
    data = get_dataset()
    X_train, X_test, y_train, y_test = Split_data(data)
    #plot_data(data, 'Gender', 'EstimatedSalary')
    train_svm(X_train, y_train, X_test, y_test, ['linear', 'poly', 'rbf'])
    
    
    
if __name__ == '__main__':
    main()