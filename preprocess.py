import sys
import pdb
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import linear_model
def mapper(sequence):
    uniqueSeq = list(sequence.unique())
    sequenceNum = [uniqueSeq.index(item)+1 for item in sequence ]
    return sequenceNum

# canonical correlation-based nan filling method
# target_column is the column with missing values in different rows
# dependent matrix is the information to infer the missing value
# I assume target_column is nx1 array, dependent matrix is n*m array, mis # of features
# In this first version, I assume dependent_matrix is all filled
def fillnan(target_column,dependent_matrix):
    # find the index of the target_column where valid value is filled
    #valid_index = [i for i in range(len(target_column)) if i not in np.isnan(target_column)]
    valid_index = np.invert(np.isnan(target_column))
    #pdb.set_trace()
    x = dependent_matrix[valid_index,:]
    y = target_column[valid_index]
    cca = CCA(n_components=3)
    cca.fit(x,y)  
    pdb.set_trace()
    cca.x_weights_

train_data = pd.read_csv('train.csv')
train_data = train_data[['Pclass','Sex','Age','Parch','SibSp','Fare','Embarked','Survived']] 
train_data = train_data.dropna()
y = train_data['Survived']
x = train_data[['Pclass','Sex','Age','Parch','SibSp','Fare','Embarked']]
x['Sex']=mapper(x['Sex'])
x['Embarked']=mapper(x['Embarked'])

decisionTree = tree.DecisionTreeClassifier()
x = x.values
y = y.values
decisionTree = decisionTree.fit(x,y)

test_data = pd.read_csv('test.csv')

test_x = test_data[['Pclass','Sex','Age','Parch','SibSp','Fare','Embarked']]
test_x['Sex']=mapper(test_x['Sex'])
test_x['Embarked']=mapper(test_x['Embarked'])
#pdb.set_trace()
test_x=test_x.values
fillnan(test_x[:,2],test_x[:,range(2)+range(3,5)+range(6,7)])
decisionTree.predict(test_x)





