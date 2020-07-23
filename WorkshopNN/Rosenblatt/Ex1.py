import numpy as np
import matplotlib.pyplot as plt

def generate_vector(mu, sigma, N):# Simple function to generate random vector from a normal distribution 
    v=np.random.normal(loc=mu, scale=sigma, size=N)
    r=generate_random()#generate label 
    v=np.hstack((v,r))
    return v


def generate_random():# Randomly generates a number(r) with 50% chance of being -1 and 50% chance of being 1 
    n=np.random.rand()
    if n > 0.5:
        r=-1
    else:
        r=1
    return r


def generate_ds(P,N):# Generates random dataset using the functions depicted above (P vectors of dimension N plus another vector for the labels)
    v1=np.zeros((P,N+1))
    for i in range(P):
        v=generate_vector(0,1,N)
        v1[i,:]=v
    return v1

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
    return np.where(prediction > 0, 1, -1)

#This is with the given paramethers (ns=50,N=20,alpha=0.75,...,3.0,nmax=100)->They are ez to change
S=np.zeros(10)
Alphas=[]
ind=0
N=20#Dimension of vectors
ns=50
for a in range(75,325,25):#alphas 
    ind=ind+1
    alpha=a*0.01
    Alphas.append(alpha)
    P=round(alpha*N)# Number of vectors 
    for i in  range(ns):
        ds=generate_ds(P,N)
        X=ds[:,:-1]
        Y=ds[:,-1:]
        y=[]
        for i in range(len(Y)):
            y.append(Y[i][0])  
        y=np.array(y)    
        Per=Perceptron(100)# Creates an object with ns=number of maximum epochs.
        Per.fit(X,y)
        if max(Per.A)==100:
            S[ind-1]=S[ind-1]+1#function to count sucsessfull runs

Succ=np.zeros(10)            
for i in range(len(S)):
    Succ[i]=S[i]/50    

    
plt.scatter(Alphas,Succ, color='darkmagenta')
plt.ylabel('Qls(%)')
plt.xlabel('Alpha')
