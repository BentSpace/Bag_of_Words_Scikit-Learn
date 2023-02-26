import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

def dt_train(X,Y):
    dt = sklearn.tree.DecisionTreeClassifier()
    dt = dt.fit(X,Y)
    return dt
 
def kmeans_train(X):
    return sklearn.cluster.KMeans(n_clusters=2).fit(X)
    
def knn_train(X,Y,K):
    return sklearn.neighbors.KNeighborsClassifier(n_neighbors=K).fit(X, Y)

def perceptron_train(X,Y):
    return sklearn.linear_model.Perceptron().fit(X, Y)
    
    
def nn_train(X,Y, hls):
    return sklearn.neural_network.MLPClassifier(hidden_layer_sizes=hls, early_stopping=False).fit(X, Y)

def pca_train(X,K):
    pca = sklearn.decomposition.PCA(n_components=K)
    return pca.fit(X)
     
def pca_transform(X,pca):
    return pca.transform(X)

def svm_train(X,Y,k):
    return sklearn.svm.SVC(kernel=k).fit(X, Y)

def model_test(X,model):
    return model.predict(X)

def compute_F1(Y, Y_hat):
    return sklearn.metrics.f1_score(Y, Y_hat)