import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay


def run():
    # Load input layer DataFrame
    MLNN_feature_layer_df = pd.read_csv('MLNN_input.csv')

    # Separate features from label
    labels = MLNN_feature_layer_df.pop('result').values
    features = MLNN_feature_layer_df.values

    # Normalize features' values by calculating the Z-scores
    features_mean = features.mean()
    features_std = features.std()
    features_norm = (features - features_mean) / features_std

    # 10-fold cross validation
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    K_fold = KFold(n_splits=10, shuffle=True, random_state=6)

    current_split = 1  # to print current split on console
    train_accuracy_list = []  # to store the train accuracy of each split
    test_accuracy_list = []  # to store the test accuracy of each split
    for train_index, test_index in K_fold.split(features_norm, labels):
        # Split data
        x_train, x_test = features_norm[train_index], features_norm[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Create a Multi-layer Perceptron classifier
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        MLP_clf = MLPClassifier(
            hidden_layer_sizes=(28, 28, 28), activation='relu', solver='adam', batch_size='auto',
            learning_rate='constant', learning_rate_init=0.001, max_iter=2000, shuffle=True, random_state=6,
            tol=0.0001, verbose=False, n_iter_no_change=10)

        # Train the model
        MLP_clf.fit(x_train, y_train)

        # Evaluate the model
        train_accuracy = MLP_clf.score(x_train, y_train)
        train_accuracy_list.append(train_accuracy)
        test_accuracy = MLP_clf.score(x_test, y_test)
        test_accuracy_list.append(test_accuracy)

        # Compute confusion matrix to evaluate the accuracy of the classification
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
        CM_train = ConfusionMatrixDisplay.from_estimator(estimator=MLP_clf, X=x_train, y=y_train,
                                                         display_labels=['Draw', 'Home', 'Away'],
                                                         cmap=plt.cm.Blues)
        CM_test = ConfusionMatrixDisplay.from_estimator(estimator=MLP_clf, X=x_test, y=y_test,
                                                        display_labels=['Draw', 'Home', 'Away'],
                                                        cmap=plt.cm.Blues)
        CM_train.ax_.set_title(f'split {current_split} : train')
        CM_test.ax_.set_title(f'split {current_split} : test')

        # Print information of current split on console
        print('---- Split ', current_split, ' ----')
        current_split += 1
        print('Train mean accuracy --> ', train_accuracy)
        print('Test mean accuracy --> ', test_accuracy)
        print()

        plt.show()

    # Print average train and test accuracy on console
    print('-- Average train accuracy : ', np.average(train_accuracy_list), ' --')
    print('-- Average test accuracy : ', np.average(test_accuracy_list), ' --')
