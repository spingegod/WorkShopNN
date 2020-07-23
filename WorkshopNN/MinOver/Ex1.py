import numpy as np
import math
import matplotlib.pyplot as plt

def generate_vector(mu, sigma, N):# Simple function to generate random vector from a normal distribution 
    v=np.random.normal(loc=mu, scale=sigma, size=N)
    return v

def generate_ds(P,N):# Generates random dataset using the functions depicted above (P vectors of dimension N plus another vector for the labels)
    v1=np.zeros((P,N))
    for i in range(P):
        v=generate_vector(0,1,N)
        v1[i,:]=v
    return v1

def generate_w(N):
    w_hat=np.random.rand(N)
    w=w_hat*math.sqrt(N)
    return w

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

N=20#Dimension of vectors
w=generate_w(N)
#This is with the given paramethers (ns=10,N=20,alpha=0.25,...,5.0,nmax=100)->They are ez to change
Alphas=[]
ind=0
Epsilons=[]
num=0
ns=2000
for a in range(10,510,10):
    alpha=a*0.01
    Alphas.append(alpha)
    P=round(alpha*N)# Number of vectors 
    E=[]
    for i in  range(10):
        X=generate_ds(P,N)
        y=[]
        for s in zip(X):
            pred=np.dot(s,w)
            pp=np.where(pred>0,1,-1)
            y.append(pp)
        y=np.array(y)   
        Per=Perceptron_MinOver(ns)# Creates an object with ns=number of maximum epochs.
        Per.fit(X,y)
        num=num+1
        print(str(500-num)+' iterations '+'to'+' end')
        epsilon=(1/np.pi)*math.acos(np.dot(Per.w,w)/(np.linalg.norm(Per.w)*np.linalg.norm(w)))#Calculating the angular change using the  teacher and stuudent's weight vectors
        E.append(epsilon)
    E=np.array(E)    
    e_mean=np.mean(E)
    Epsilons.append(e_mean)
Epsilons=np.array(Epsilons)    
print('finished!!!')

plt.scatter(Alphas,Epsilons, color='darkmagenta')
plt.ylabel('Angular change (teta)')
plt.xlabel('Alpha')

