import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

# Assignment 1: PCA
class PCA():
    def __init__(self, X):           
        #center the data   
        self.mean = X.mean(axis=0)
        X = X - self.mean
        #calculate covariance matrix based on X where data points are represented in rows
        C = np.cov(X, rowvar=False)    
        #get eigenvectors and eigenvalues
        d,u = np.linalg.eigh(C)        
        #sort both eigenvectors and eigenvalues descending regarding the eigenvalue
        #the output of np.linalg.eigh is sorted ascending
        #therefore both are turned around to reach a descending order
        self.U = np.asarray(u).T[::-1]  
        self.D = d[::-1]
    
    def project(self, X, m):
        #again data needs to be centered
        X = X - self.mean
        #use the top m eigenvectors with the highest eigenvalues for the transformation matrix
        Z = np.dot(X,np.asmatrix(self.U[:m]).T)
        return Z
    
    def denoise(self, X, m):
        #check for the amount of eigenvectors
        if m < 1 or m > len(X):
            raise ValueError('Number of eigenvectors (m) not within the range.\n')
        #call project-function
        Z = PCA.project(self, X, m)
        #matrix multiplication of the m eigenvectors and the Z matrix
        X_denoised = np.dot(self.U[:m].T,Z.T).T
        #decenter the data
        X_denoised = X_denoised+self.mean
        return X_denoised
    
