from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding

import numpy as np
def iris():
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
    return X_train, X_test
    # Evaluate the clusters using silhouette score
    kmeans_silhouette = silhouette_score(X, kmeans_labels).round(2)


    print(f'Silhouette Score for KMeans: {kmeans_silhouette}')

    # Visualizing the clusters (just picking the first two features for simplicity)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title('K-Means Clustering')

    plt.show()

def weather():

    df = pd.read_csv('train_ori.csv')

    # Let's assume the dataset needs to have missing values handled and then only numeric columns are kept
    df = df.dropna()  # Or other more sophisticated missing value handling
    df_numeric = df
    # Scale the data to normalize it
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)
    X_train, X_test = train_test_split(X_scaled, test_size=0.5, random_state=42)
    return X_train, X_test
    # print(X_scaled )
    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)  # You need to choose the appropriate number of clusters
    kmeans_labels = kmeans.fit_predict(X_train)

    # Evaluate the clusters using silhouette score
    kmeans_silhouette = silhouette_score(X_train, kmeans_labels).round(2)
    print(f'Silhouette Score for KMeans: {kmeans_silhouette}')

    # Visualizing the clusters - pick the first two numeric columns for a 2D plot
    plt.figure(figsize=(12, 6))

    plt.scatter(X_train[:, 0], X_train[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title('K-Means Clustering')
    plt.show()

def train_Kmeans(X,type,nn):
    kmeans = KMeans(n_clusters=nn, random_state=42)  # You need to choose the appropriate number of clusters
    kmeans_labels = kmeans.fit_predict(X)

    # Evaluate the clusters using silhouette score
    kmeans_silhouette = silhouette_score(X, kmeans_labels).round(2)
    print(f'Silhouette Score for KMeans: {kmeans_silhouette}')

    plt.figure(figsize=(12, 6))

    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title('K-Means Clustering')
    # plt.show()
    plt.savefig(f'{type}.png')
    return kmeans_silhouette
def train_em(X,type,nn):
    em = GaussianMixture(n_components=nn, random_state=42)  # Again, choose the appropriate number of components
    em_labels = em.fit_predict(X)

    # Evaluate the clusters using silhouette score
    em_silhouette = silhouette_score(X, em_labels).round(2)

    print(f'Silhouette Score for EM: {em_silhouette}')

    # Visualizing the clusters - pick the first two numeric columns for a 2D plot
    plt.figure(figsize=(12, 6))

    plt.scatter(X[:, 0], X[:, 1], c=em_labels, cmap='viridis')
    plt.title(f'EM Clustering_{type}')
    plt.savefig(f'./plots/{type}.png')
    # plt.show()
    return em_silhouette

    
if __name__ == "__main__":
    X,_ = iris()
    ss = []
    ss.append(train_Kmeans(X,'1-1',2))
    ss.append(train_em(X,'1-2',2))
    
    X,_ = weather()
    ss.append(train_Kmeans(X,'1-3',3))
    ss.append(train_em(X,'1-4',3))
    
    print('iris_Kmeans_silhouette_score:',ss[0])
    print('iris_em_silhouette_score:',ss[1])
    print('weather_Kmeans_silhouette_score:',ss[2])
    print('weather_em_silhouette_score:',ss[3])
