import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
#import seaborn as sns


df = pd.read_csv("/home/welcome/Downloads/ml_ai_dl/Pizza.csv", usecols = ["brand", "mois", "prot", "fat", "ash", "sodium", "carb", "cal"])
#print(df)
correlation = df.corr()
#print(correlation)
#print(correlation.shape)

X = df.iloc[:,1:8].values
#print(X)
#print(X.shape)
y = df.iloc[:,0].values
print(y)
print(y.shape)

X_mean = np.mean(X, axis = 0)
#print(X_mean)
#print(X_mean.shape)
X_deviation = np.std(X, axis =0)
Z = (X - X_mean)/X_deviation
#print(Z)
print(Z.shape)

cov_mat = np.cov(Z.transpose())
print(Z.shape)
#print(cov_mat)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#print(eig_vals)
#print(eig_vecs)
eig_pairs = [(eig_vals[i], eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()
#print(eig_pairs)


feature_vector = [eig_pairs[i][1] for i in range(0,3)]
feature_vector = np.asarray(feature_vector).transpose()
#print(feature_vector)

updated_X = np.dot(Z,feature_vector)
print(updated_X)




