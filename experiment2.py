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
    print(df_numeric)
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
    # plt.savefig(f'./plots/{type}.png')
    plt.savefig('1-2-2.png')
    return kmeans_silhouette

def train_Em(X,type,nn):
    em = GaussianMixture(n_components=3, random_state=42)  # Again, choose the appropriate number of components
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
import seaborn as sns
from scipy.stats import kurtosis
if __name__ == "__main__":
    X,_ = iris()
    # pca = PCA().fit(X) 
    # # Plotting the Cumulative Summation of the Explained Variance
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Variance (%)') # for each component
    # plt.title('Explained Variance')
    # plt.grid(True)
    # plt.savefig('2-1-1-1.png')

    # iris = datasets.load_iris()
    # iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # X_iris = iris_df.values
    # y_iris = pd.Categorical.from_codes(iris.target, iris.target_names)
    # scaler = StandardScaler()
    # X_iris_scaled = scaler.fit_transform(X_iris)
    # pca = PCA(n_components=2)  # Reduce to 2 principal components
    # X_iris_pca = pca.fit_transform(X_iris_scaled)
    # iris_pca_df = pd.DataFrame(X_iris_pca, columns=['PCA1', 'PCA2'])
    # iris_pca_df['species'] = y_iris
    # sns.pairplot(iris_pca_df, hue='species', plot_kws={'alpha': 0.7})  # Adjust alpha for point transparency
    # plt.savefig('2-1-2.png')
    # plt.close()
    
    # # Dimensionality reduction with ICA
    # ica = FastICA(n_components=3)  # Adjust n_components for your analysis
    # X_ica = ica.fit_transform(X)
    # n_components_range = range(1, X.shape[1] + 1)
    # kurtosis_values = []
    # for n_components in n_components_range:
    #     ica = FastICA(n_components=n_components, random_state=42)
    #     ica.fit(X)
    #     components = ica.transform(X)
    #     component_kurtosis = kurtosis(components, fisher=True, axis=0)
    #     kurtosis_values.append(component_kurtosis)
    # mean_absolute_kurtosis = [np.mean(np.abs(k)) for k in kurtosis_values]
    # optimal_n_components = np.argmax(mean_absolute_kurtosis) + 1  # Adding 1 because index starts from 0
    # print(f'Optimal number of components based on kurtosis: {optimal_n_components}')
    # plt.figure(figsize=(10, 5))
    # plt.plot(n_components_range, mean_absolute_kurtosis, marker='o')
    # plt.title('ICA Component Selection')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Mean Absolute Kurtosis')
    # plt.grid(True)
    # plt.xticks(n_components_range)
    # plt.savefig('2-1-2-1.png')

    # iris = datasets.load_iris()
    # iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # X_iris = iris_df.values
    # y_iris = pd.Categorical.from_codes(iris.target, iris.target_names)
    # scaler = StandardScaler()
    # X_iris_scaled = scaler.fit_transform(X_iris)
    # ica = FastICA(n_components=2)  # Reduce to 2 principal components
    # X_ica = ica.fit_transform(X_iris_scaled)
    # iris_pca_df = pd.DataFrame(X_ica, columns=['PCA1', 'PCA2'])
    # iris_pca_df['species'] = y_iris
    # sns.pairplot(iris_pca_df, hue='species', plot_kws={'alpha': 0.7})  # Adjust alpha for point transparency
    # plt.savefig('2-1-2-2.png')
    # plt.close()

    # # Dimensionality reduction with Randomized Projections
    # random_projection = GaussianRandomProjection(n_components=3)  # Adjust n_components for your analysis
    # X_random_proj = random_projection.fit_transform(X)

    # reconstruction_errors = []
    # n_components_range = range(1, X.shape[1] + 1)  # Range of component numbers to consider
    # for n_components in n_components_range:
    #     grp = GaussianRandomProjection(n_components=n_components)
    #     X_projected = grp.fit_transform(X)
    #     X_reconstructed = np.dot(X_projected, np.linalg.pinv(grp.components_.T))
    #     reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    #     reconstruction_errors.append(reconstruction_error)
    # # Plot the reconstruction error as a function of the number of components
    # plt.figure(figsize=(10, 5))
    # plt.plot(n_components_range, reconstruction_errors, marker='o')
    # plt.title('Random Projection Reconstruction Error')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Reconstruction Error')
    # plt.grid(True)
    # plt.xticks(n_components_range)
    # plt.savefig('2-1-3-1.png')#33333

    # iris = datasets.load_iris()
    # iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # X_iris = iris_df.values
    # y_iris = pd.Categorical.from_codes(iris.target, iris.target_names)
    # scaler = StandardScaler()
    # X_iris_scaled = scaler.fit_transform(X_iris)
    # random_projection = GaussianRandomProjection(n_components=3)  
    # X_ica = random_projection.fit_transform(X_iris_scaled)
    # iris_pca_df = pd.DataFrame(X_ica, columns=['PCA1', 'PCA2','PCA3'])
    # iris_pca_df['species'] = y_iris
    # sns.pairplot(iris_pca_df, hue='species', plot_kws={'alpha': 0.7})  # Adjust alpha for point transparency
    # plt.savefig('2-1-3-2.png')
    # plt.close()

    # lle = LocallyLinearEmbedding(n_components=3, n_neighbors=12, random_state=42)  # Adjust parameters as needed
    # X_lle = lle.fit_transform(X)
    # reconstruction_errors = []
    # n_neighbors = 10  # typically set to a constant value that is suitable for your dataset
    # n_components_range = range(1, X.shape[1])  # trying different values for n_components

    # for n_components in n_components_range:
    #     lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, random_state=42, method='standard')
    #     X_reduced = lle.fit_transform(X)
    #     reconstruction_errors.append(lle.reconstruction_error_)

    # # Plot the reconstruction error as a function of the number of components
    # plt.figure(figsize=(10, 5))
    # plt.plot(n_components_range, reconstruction_errors, marker='o')
    # plt.title('LLE Reconstruction Error')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Reconstruction Error')
    # plt.grid(True)
    # plt.xticks(n_components_range)
    # plt.savefig('2-1-4-1.png')#222
    # iris = datasets.load_iris()
    # iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # X_iris = iris_df.values
    # y_iris = pd.Categorical.from_codes(iris.target, iris.target_names)
    # scaler = StandardScaler()
    # X_iris_scaled = scaler.fit_transform(X_iris)
    # lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
    # X_ica = lle.fit_transform(X_iris_scaled)
    # iris_pca_df = pd.DataFrame(X_ica, columns=['PCA1', 'PCA2'])
    # iris_pca_df['species'] = y_iris
    # sns.pairplot(iris_pca_df, hue='species', plot_kws={'alpha': 0.7})  # Adjust alpha for point transparency
    # plt.savefig('2-1-4-2.png')
    # plt.close()


    X,_ = weather()

    # pca = PCA(n_components=2)  # Adjust n_components for your analysis
    # X_pca = pca.fit_transform(X)
    # # Plotting the Cumulative Summation of the Explained Variance
    # pca = PCA().fit(X) 
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Variance (%)') # for each component
    # plt.title('Explained Variance')
    # plt.grid(True)
    # plt.savefig('2-1-5-1.png') #4444
    # plt.close()

    # weather_df = pd.read_csv('train_ori.csv')
    # weather_df['high_humidity_label'] = (weather_df['relative_humidity_3pm'] > 24.99) * 1
    # weather_df.drop('relative_humidity_3pm', axis=1, inplace=True)
    # X_weather = weather_df.drop('high_humidity_label', axis=1)
    # y_weather = weather_df['high_humidity_label']
    # scaler = StandardScaler()
    # X_weather_scaled = scaler.fit_transform(X_weather)
    # pca = PCA(n_components=4)  # Adjust n_components as necessary
    # X_weather_pca = pca.fit_transform(X_weather_scaled)
    # weather_pca_df = pd.DataFrame(X_weather_pca, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4'])
    # weather_pca_df['high_humidity_label'] = y_weather
    # sns.pairplot(weather_pca_df, hue='high_humidity_label', plot_kws={'alpha': 0.7})
    # plt.savefig('2-1-5-2.png')
    # plt.close()

    # # Dimensionality reduction with ICA
    # ica = FastICA(n_components=2)  # Adjust n_components for your analysis
    # X_ica = ica.fit_transform(X)
    # n_components_range = range(1, X.shape[1] + 1)
    # kurtosis_values = []
    # for n_components in n_components_range:
    #     ica = FastICA(n_components=n_components, random_state=42)
    #     ica.fit(X)
    #     components = ica.transform(X)
    #     component_kurtosis = kurtosis(components, fisher=True, axis=0)
    #     kurtosis_values.append(component_kurtosis)
    # mean_absolute_kurtosis = [np.mean(np.abs(k)) for k in kurtosis_values]
    # optimal_n_components = np.argmax(mean_absolute_kurtosis) + 1  # Adding 1 because index starts from 0
    # print(f'Optimal number of components based on kurtosis: {optimal_n_components}')
    # plt.figure(figsize=(10, 5))
    # plt.plot(n_components_range, mean_absolute_kurtosis, marker='o')
    # plt.title('ICA Component Selection')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Mean Absolute Kurtosis')
    # plt.grid(True)
    # plt.xticks(n_components_range)### 7777
    # plt.savefig('2-1-6-1.png')

    # weather_df = pd.read_csv('train_ori.csv')
    # weather_df['high_humidity_label'] = (weather_df['relative_humidity_3pm'] > 24.99) * 1
    # weather_df.drop('relative_humidity_3pm', axis=1, inplace=True)
    # X_weather = weather_df.drop('high_humidity_label', axis=1)
    # y_weather = weather_df['high_humidity_label']
    # scaler = StandardScaler()
    # X_weather_scaled = scaler.fit_transform(X_weather)
    # ica = FastICA(n_components=7)  # Adjust n_components as necessary
    # X_weather_pca = ica.fit_transform(X_weather_scaled)
    # weather_pca_df = pd.DataFrame(X_weather_pca, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4','PCA5','PCA6','PCA7'])
    # weather_pca_df['high_humidity_label'] = y_weather
    # sns.pairplot(weather_pca_df, hue='high_humidity_label', plot_kws={'alpha': 0.7})
    # plt.savefig('2-1-6-2.png')
    # plt.close()

    # # Dimensionality reduction with Randomized Projections
    # random_projection = GaussianRandomProjection(n_components=2)  # Adjust n_components for your analysis
    # X_random_proj = random_projection.fit_transform(X)
    # reconstruction_errors = []
    # n_components_range = range(1, X.shape[1] + 1)  # Range of component numbers to consider
    # for n_components in n_components_range:
    #     grp = GaussianRandomProjection(n_components=n_components)
    #     X_projected = grp.fit_transform(X)
    #     X_reconstructed = np.dot(X_projected, np.linalg.pinv(grp.components_.T))
    #     reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    #     reconstruction_errors.append(reconstruction_error)
    # # Plot the reconstruction error as a function of the number of components
    # plt.figure(figsize=(10, 5))
    # plt.plot(n_components_range, reconstruction_errors, marker='o')
    # plt.title('Random Projection Reconstruction Error')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Reconstruction Error')
    # plt.grid(True)
    # plt.xticks(n_components_range)
    # plt.savefig('2-1-7-1.png')#7777
    # weather_df = pd.read_csv('train_ori.csv')
    # weather_df['high_humidity_label'] = (weather_df['relative_humidity_3pm'] > 24.99) * 1
    # weather_df.drop('relative_humidity_3pm', axis=1, inplace=True)
    # X_weather = weather_df.drop('high_humidity_label', axis=1)
    # y_weather = weather_df['high_humidity_label']
    # scaler = StandardScaler()
    # X_weather_scaled = scaler.fit_transform(X_weather)
    # ica = GaussianRandomProjection(n_components=7)  # Adjust n_components as necessary
    # X_weather_pca = ica.fit_transform(X_weather_scaled)
    # weather_pca_df = pd.DataFrame(X_weather_pca, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4','PCA5','PCA6','PCA7'])
    # weather_pca_df['high_humidity_label'] = y_weather
    # sns.pairplot(weather_pca_df, hue='high_humidity_label', plot_kws={'alpha': 0.7})
    # plt.savefig('2-1-7-2.png')
    # plt.close()

    # lle = LocallyLinearEmbedding(n_components=8, n_neighbors=12, random_state=42)  # Adjust parameters as needed
    # X_lle = lle.fit_transform(X,2)
    reconstruction_errors = []
    n_neighbors = 12  # typically set to a constant value that is suitable for your dataset
    n_components_range = range(1, X.shape[1])  # trying different values for n_components

    for n_components in n_components_range:
        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, random_state=42, method='standard')
        X_reduced = lle.fit_transform(X)
        reconstruction_errors.append(lle.reconstruction_error_)

    # Plot the reconstruction error as a function of the number of components
    plt.figure(figsize=(10, 5))
    plt.plot(n_components_range, reconstruction_errors, marker='o')
    plt.title('LLE Reconstruction Error')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.grid(True)
    plt.xticks(n_components_range)
    plt.savefig('2-1-8-1.png')#222

    weather_df = pd.read_csv('train_ori.csv')
    weather_df['high_humidity_label'] = (weather_df['relative_humidity_3pm'] > 24.99) * 1
    weather_df.drop('relative_humidity_3pm', axis=1, inplace=True)
    X_weather = weather_df.drop('high_humidity_label', axis=1)
    y_weather = weather_df['high_humidity_label']
    scaler = StandardScaler()
    X_weather_scaled = scaler.fit_transform(X_weather)
    ica = LocallyLinearEmbedding(n_components=7, n_neighbors=12, random_state=42)  # Adjust n_components as necessary
    X_weather_pca = ica.fit_transform(X_weather_scaled)
    weather_pca_df = pd.DataFrame(X_weather_pca, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4','PCA5','PCA6','PCA7'])
    weather_pca_df['high_humidity_label'] = y_weather
    sns.pairplot(weather_pca_df, hue='high_humidity_label', plot_kws={'alpha': 0.7})
    plt.savefig('2-1-8-2.png')
    plt.close()
