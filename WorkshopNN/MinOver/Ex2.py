import numpy as np
import math
import pandas as pd
import sklearn.preprocessing as skp
from random import randint

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

class Perceptron_MinOver:# Code to generate objects from the Perceptron class using MinOver algortihm for stability

    
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
      K=[]
      L=[]
      for s, y_true in zip(X, y):  
        k=np.dot(self.w,s)*y_true/np.linalg.norm(self.w)#Computes stability
        K.append(k)
        y_pred    = self.pred(s)
        d = (y_true - y_pred)
        L.append(d)
        acc=self.accuracy(L)*100
      #print('Epoch: ' + str(e) + ' ; ' + 'Fitting accuracy: ' + str(round(acc,2)) + ' %')
      i_min = min(range(len(K)), key=K.__getitem__)#Checks index of feature vector with minimal stability
      ac = ((1/np.pi)*math.acos(np.dot(self.w,self.w + (1/n_f)*X[i_min]*y[i_min])/(np.linalg.norm(self.w)*np.linalg.norm(self.w + (1/n_f)*X[i_min]*y[i_min]))))
      if ac<0.005:#Here optimal stability is considered if the angular change between updates is smaller than 0.005 ()
          print('Optimal Stability was acheived in '+str(e)+' epochs')
          break
      self.w=self.w + (1/n_f)*X[i_min]*y[i_min]
      self.A.append(acc)
      E.append(e)
    if i==self.ns-1:
        print("The maximum number of epochs was reached")  
    return self
      
      
  def accuracy(self, D):# Computes accuracy in a given epoch
      N=[]
      for i in range(len(D)):
          if D[i]==0:
              N.append(1)
      acc=len(N)/len(D)
      return acc
      
      
  def pred(self, s):# Makes prediction using the dot product between w and x
    prediction = np.dot(s, self.w)
    return np.where(prediction > 0, 1, -1)

ds = pd.read_csv('pulsar_stars.csv').values
full_ds=ds[:,0:8]
full_ds_norm = skp.normalize(full_ds,axis=0)
full_labels=ds[:,8]
#Imports the dataset


from sklearn.model_selection import train_test_split
train_norm, X_test, labels_train_norm, y_test_o = train_test_split(full_ds_norm, full_labels, test_size=0.05, random_state=0)
#Splits the dataset into training(95%) and test(5%)


V=undersampling(train_norm,labels_train_norm)#UnderSampling applied in the training data

X=V[0]
y_o=V[1]

y=[]
for i in y_o:
    if i==1:
        y.append(1)
    else:
        y.append(-1)
y=np.array(y)

y_test=[]
for i in y_test_o:
    if i==1:
        y_test.append(1)
    else:
        y_test.append(-1)
y_test=np.array(y_test)

Per=Perceptron_MinOver(1000)#Creates an object of the class Perceptron with the maximum number of epochs=1000
Per.fit(X,y)#Iterates through feature vectors and trains the model
y_pred=Per.pred(X_test)#Predicts the test labels
ACC=check_accuracy(y_pred, y_test)#Computes accuracy
PREC=precision(y_pred, y_test)#Precision
REC=recall(y_pred,y_test)#Recall
F1=F1score(PREC,REC)#F1 Score

print('MinOver Perceptron accuracy is: ', ACC)
print('MinOver Perceptron precision is: ', PREC)
print('MinOver Perceptron recall is: ', REC)
print('MinOver Perceptron F1 score is: ', F1)