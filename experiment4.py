from mm.nn_module import nn_algorithm

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding


def main():

    # Loading and cleaning data
    data1 = pd.read_csv('./train_ori.csv')
    data2 = pd.read_csv('./test_ori.csv')

    # Creating target column for high humidity
    data1['high_humidity_label'] = (data1['relative_humidity_3pm'] > 24.99) * 1
    data2['high_humidity_label'] = (data2['relative_humidity_3pm'] > 24.99) * 1
    # Features for prediction
    morning_features = ['air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am',
                        'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am',
                        'rain_accumulation_9am', 'rain_duration_9am', 'relative_humidity_9am']

    # Splitting data
    X_train, X_test, y_train, y_test = data1[morning_features],data2[morning_features],data1['high_humidity_label'],data2['high_humidity_label']

    # Call decision tree algorithm
    # decision_tree_accuracy = decision_tree_algorithm(X_train, y_train, X_test, y_test)

    # Call neural network algorithm

    # neural_network_accuracy = nn_algorithm(X_train, y_train, X_test, y_test,10,'baseline')
    # pca = PCA(n_components=4)  # Adjust n_components for your analysis
    # X_pca = pca.fit_transform(X_train)
    # X_pca_test = pca.transform(X_test)
    # neural_network_accuracy1 = nn_algorithm(X_pca, y_train, X_pca_test, y_test,4,'4-1')
    
    # ica = FastICA(n_components=7)  # Adjust n_components for your analysis
    # X_ica = ica.fit_transform(X_train)
    # X_ica_test = ica.transform(X_test)
    # neural_network_accuracy2 = nn_algorithm(X_ica, y_train, X_ica_test, y_test,7,'4-2')
    # # Dimensionality reduction with Randomized Projections

    # random_projection = GaussianRandomProjection(n_components=7)  # Adjust n_components for your analysis
    # X_random_proj = random_projection.fit_transform(X_train)
    # X_random_proj_test = random_projection.transform(X_test)
    # neural_network_accuracy3 = nn_algorithm(X_random_proj, y_train, X_random_proj_test, y_test,7,'4-3')

    lle = LocallyLinearEmbedding(n_components=7, n_neighbors=12, random_state=42)  # Adjust parameters as needed
    X_lle = lle.fit_transform(X_train)
    X_lle_test = lle.transform(X_test)
    neural_network_accuracy4 = nn_algorithm(X_lle, y_train, X_lle_test, y_test,7,'4-4')

    # Call boosting algorithm
    # boosting_accuracy = boosting_algorithm(X_train, y_train, X_test, y_test)

    # Call SVM algorithm    # svm_accuracy = svm_algorithm(X_train, y_train, X_test, y_test)

    # Call k-NN algorithm
    # knn_accuracy = knn_algorithm(X_train, y_train, X_test, y_test)


    # print(neural_network_accuracy)

    # algorithms = ['Decision Tree', 'Neural Network', 'Boosting', 'SVM', 'k-NN']
    # accuracies = [decision_tree_accuracy, neural_network_accuracy, boosting_accuracy, svm_accuracy, knn_accuracy]
    # algorithms = ['Neural Network','PCA','ICA','Randomized Projections','LLE']
    # accuracies = [ neural_network_accuracy,neural_network_accuracy1,neural_network_accuracy2,neural_network_accuracy3,neural_network_accuracy4]
    # colors = ['blue', 'green', 'red', 'purple', 'orange']


    # plt.figure(figsize=(10, 6))
    # bars = plt.bar(algorithms, accuracies, color=colors, width=0.6)
    # plt.ylim([min(accuracies) - 0.05, max(accuracies) + 0.05])

    # # Label with the exact accuracy above each bar
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 2), ha='center', va='bottom',
    #              color='black')

    # plt.xlabel('Algorithms')
    # plt.ylabel('Accuracy')
    # plt.title('Comparison of Different Algorithms')
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
