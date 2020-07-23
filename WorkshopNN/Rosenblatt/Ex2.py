import numpy as np
import pandas as pd
import sklearn.preprocessing as skp
from random import randint

#Undersampling when the minority class is 0.
def undersampling(train,labels):
    n=0
    Ind_pulsar=[]
    Ind_not=[]
    for i in range(len(labels)):
        if labels[i]==1:
            Ind_pulsar.append(n)
        else:
            Ind_not.append(n)
        n=n+1
    while len(Ind_not)>len(Ind_pulsar):
        r=randint(0,len(Ind_not)-1)
        del Ind_not[r]
    ind=0
    I=[]
    for i in range(len(labels)):
        I.append(ind)
        ind=ind+1
    for i in range(len(Ind_not)):
        I.remove(Ind_not[i])
    for i in range(len(Ind_pulsar)):
        I.remove(Ind_pulsar[i])
        
    X=np.delete(train,I,0)
    y=np.delete(labels,I,0)
    return [X,y]

#Metrics:
#Precision the number of true positives divided by all positive predictions. 
#Recall is the number of true positives divided by the number of positive 
#values in the test data.
#F1 score the weighted average of precision and recall.

def check_accuracy(y_pred,y_true):
    n=0
    for i in range(len(y_true)):
        if y_pred[i]==y_true[i]:
            n=n+1
    acc = n/len(y_true)
    return acc

def precision(pred,test):
    true_positives = 0
    positive_predictions = (pred == 1).sum()
    for i in range(0,len(pred)):
        if pred[i] == test[i] and test[i] == 1:
            true_positives+=1
    return true_positives/positive_predictions
    
def recall(pred,test):
    true_positives = 0
    positives = (test == 1).sum()
    for i in range(0,len(pred)):
        if pred[i] == test[i] and test[i] == 1:
            true_positives+=1
    return true_positives/positives

def F1score(precision,recall):
    score = 2*(precision*recall)/(precision+recall)
    return score

class Perceptron:# Code to generate objects from the Perceptron class

    
  def __init__(self, ns):# Constructor
    self.ns = ns# Number of epochs
    

  def fit(self, X, y):# Training the Perceptron
    n_f = X.shape[1]
    self.w = np.zeros(n_f)# Initialization of weights vector with zeros 
    e=0
    E=[]
    self.A=[]
    for i in range(self.ns):# Starting the iterations through epochs
      e=e+1
      L=[]
      for s, y_true in zip(X, y):
        y_pred    = self.pred(s)# Make prediction
        d    = (y_true - y_pred)# Integer that defines if a label was predicted correctly or not (d=0 if it is a correct prediction or d=+-1 if its not)
        if (d == 2)or(d == -2):
            d==1
        w_new = d  # Compute weight update via Perceptron Learning Rule (equivalent to product of the dot product between sample and weight vector and true label)
        self.w    += (1/n_f)*w_new * s
        L.append(d)
        acc=self.accuracy(L)*100# Compute accuracy 
      self.A.append(acc)
      E.append(e)
      if acc==100:
          break# Breaking loop if 100% accuracy is reached 
    return self  
      
  def accuracy(self, D):# Computes accuracy in a given epoch
      N=[]
      for i in range(len(D)):
          if D[i]==0:
              N.append(1)#N counts the amount of labels guessed right
      acc=len(N)/len(D)
      return acc
    
  def pred(self, s):# Makes prediction using the dot product between w and x
    prediction = np.dot(s, self.w)
    return np.where(prediction > 0, 1, 0)

ds = pd.read_csv('pulsar_stars.csv').values
full_ds=ds[:,0:8]
full_ds_norm = skp.normalize(full_ds,axis=0)
full_labels=ds[:,8]

#Imports the dataset

from sklearn.model_selection import train_test_split
train_norm, X_test, labels_train_norm, y_test = train_test_split(full_ds_norm, full_labels, test_size=0.05, random_state=0)
#Splits the dataset into training(95%) and test(5%)


V=undersampling(train_norm,labels_train_norm)#UnderSampling applied in the training data

X=V[0]
y=V[1]

Per=Perceptron(100)#Creates an object of the class Perceptron with the maximum number of epochs=100
Per.fit(X,y)#Iterates through feature vectors and trains the model
y_pred=Per.pred(X_test)#Predicts the test labels
ACC=check_accuracy(y_pred, y_test)#Computes accuracy
PREC=precision(y_pred, y_test)#Precision
REC=recall(y_pred,y_test)#Recall
F1=F1score(PREC,REC)#F1 Score

print('Rosenblats Perceptron accuracy is: ', ACC)
print('Rosenblats Perceptron precision is: ', PREC)
print('Rosenblats Perceptron recall is: ', REC)
print('Rosenblats Perceptron F1 score is: ', F1)


