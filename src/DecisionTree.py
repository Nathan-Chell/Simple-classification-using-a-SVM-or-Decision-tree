#Preform linear classification on a dataset using a desicion tree

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn import tree

import graphviz
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

def plot_tree(model, X_train, y_train):
        
    dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=X_train.columns.values[:4],  
                                class_names=y_train.columns.values[0],
                                filled=True)

    # Draw graph
    graph = graphviz.Source(dot_data, format="png") 
    graph.render("DecisionTree")

    
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

def train_decision_tree(X_train, y_train, X_test, y_test):

    #train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train.values.ravel())
    
    #Calculate the Confusion matrix
    #print("Confusion matrix: {}".format(metrics.confusion_matrix(y_test, model.predict(X_test))))
    
    #Calculate the accuracy of the model
    print("Model accuracy: {}".format(metrics.accuracy_score(y_test, model.predict(X_test))))
    
    return model

def main():
    
    data = get_dataset()
    X_train, X_test, y_train, y_test = Split_data(data)
    #plot_data(data, 'Gender', 'EstimatedSalary')
    
    model = train_decision_tree(X_train, y_train, X_test, y_test)
    plot_tree(model, X_train, y_train)
    
    
    
if __name__ == '__main__':
    main()