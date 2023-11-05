from mm.nn_module import nn_algorithm

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding


def main1():
    ffname = ['em','kmeans']
    ff = [['test_em.csv','train_em.csv'],['test_kmeans.csv','train_kmeans.csv']]
    for i in ff:
        # Loading and cleaning data
        # data = pd.read_csv(f'./{i}')
        # data = data.dropna()
        testset = pd.read_csv(f'./{i[0]}')
        trainset = pd.read_csv(f'./{i[1]}')
        # Creating target column for high humidity
        # data['high_humidity_label'] = (data['relative_humidity_3pm'] > 24.99) * 1
        testset['high_humidity_label'] = (testset['relative_humidity_3pm'] > 24.99) * 1
        trainset['high_humidity_label'] = (trainset['relative_humidity_3pm'] > 24.99) * 1
        # print(data['high_humidity_label'])
        print(trainset['Labels'])
        # Features for prediction
        morning_features = ['air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am',
                            'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am',
                            'rain_accumulation_9am', 'rain_duration_9am', 'relative_humidity_9am']

        # X = data[morning_features]
        # y = data['high_humidity_label']

        # # Splitting data
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324,)
        X_train = trainset[morning_features]
        y_train = trainset['high_humidity_label']
        X_test = testset[morning_features]
        y_test = testset['high_humidity_label']

        neural_network_accuracy = nn_algorithm(X_train, y_train, X_test, y_test,10,'baseline')




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
def main2():
    ffname = ['em','kmeans']
    ff = [['test_em.csv','train_em.csv'],['test_kmeans.csv','train_kmeans.csv']]
    for i in ff:
        # Loading and cleaning data
        # data = pd.read_csv(f'./{i}')
        # data = data.dropna()
        testset = pd.read_csv(f'./{i[0]}')
        trainset = pd.read_csv(f'./{i[1]}')
        # Creating target column for high humidity
        # data['high_humidity_label'] = (data['relative_humidity_3pm'] > 24.99) * 1
        testset['high_humidity_label'] = (testset['relative_humidity_3pm'] > 24.99) * 1
        trainset['high_humidity_label'] = (trainset['relative_humidity_3pm'] > 24.99) * 1
        # print(data['high_humidity_label'])
        print(trainset['Labels'])
        # Features for prediction
        morning_features = ['air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am',
                            'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am',
                            'rain_accumulation_9am', 'rain_duration_9am', 'relative_humidity_9am','Labels']

        # X = data[morning_features]
        # y = data['high_humidity_label']

        # # Splitting data
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324,)
        X_train = trainset[morning_features]
        y_train = trainset['high_humidity_label']
        X_test = testset[morning_features]
        y_test = testset['high_humidity_label']

        neural_network_accuracy = nn_algorithm(X_train, y_train, X_test, y_test,10,f'add_feature{i}')




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
    main1()
    main2()
