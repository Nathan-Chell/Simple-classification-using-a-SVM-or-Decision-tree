#Preform linear classification on a dataset using a desicion tree

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn import preprocessing



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

def plot_tree(model, X_train, y_train, title):
        
    dot_data = export_graphviz(model, out_file=None, 
                                feature_names=X_train.columns.values[:4],  
                                class_names=y_train.columns.values[0],
                                filled=True)

    # Draw graph
    graph = graphviz.Source(dot_data, format="png") 
    graph.render(title)

    
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
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f'Train score {accuracy_score(y_train_pred,y_train)}')
    print(f'Test score {accuracy_score(y_test_pred,y_test)}')
    
    plot_tree(model, X_train, y_train, 'Decision_Tree')

def train_decision_tree_with_pruning(X_train, y_train, X_test, y_test):
    
    #Determine the optimal alpha value for pruning
    
    
    #TempModel = DecisionTreeClassifier(random_state=0)
    
    #path = TempModel.cost_complexity_pruning_path(X_train, y_train)
    #ccp_alphas = path.ccp_alphas
    
    #models = []
    #for ccp_alpha in ccp_alphas:
        #clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        #clf.fit(X_train, y_train)
        #models.append(clf)
        
    #models = models[:-1]
    #ccp_alphas = ccp_alphas[:-1]
     
    #train_acc = []
    #test_acc = []
    #for c in models:
        #y_train_pred = c.predict(X_train)
        #y_test_pred = c.predict(X_test)
        #train_acc.append(accuracy_score(y_train_pred,y_train))
        #test_acc.append(accuracy_score(y_test_pred,y_test))
    
    #Plot the accuracy vs alpha
    #From this plot an alpha of 0.006 is chosen
    
    #plt.scatter(ccp_alphas,train_acc)
    #plt.scatter(ccp_alphas,test_acc)
    #plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
    #plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
    #plt.legend()
    #plt.title('Accuracy vs alpha')
    #plt.show()
    
    
    clf_ = DecisionTreeClassifier(random_state=0,ccp_alpha=0.006)
    clf_.fit(X_train,y_train)
    y_train_pred = clf_.predict(X_train)
    y_test_pred = clf_.predict(X_test)

    print(f'Train score {accuracy_score(y_train_pred,y_train)}')
    print(f'Test score {accuracy_score(y_test_pred,y_test)}')
    
    plot_tree(clf_, X_train, y_train, 'Pruned_Decision_Tree')
    
def main():
    
    data = get_dataset()
    X_train, X_test, y_train, y_test = Split_data(data)
    #plot_data(data, 'Gender', 'EstimatedSalary')
    
    #train_decision_tree(X_train, y_train, X_test, y_test)
    train_decision_tree_with_pruning(X_train, y_train, X_test, y_test)
    
    
    
if __name__ == '__main__':
    main()