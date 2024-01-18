import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def run():
    # Load 4 DataFrames with the odds of each betting company
    B365_odds_df = pd.read_csv('B365_odds.csv')[['B365H', 'B365D', 'B365A']]
    BW_odds_df = pd.read_csv('BW_odds.csv')[['BWH', 'BWD', 'BWA']]
    IW_odds_df = pd.read_csv('IW_odds.csv')[['IWH', 'IWD', 'IWA']]
    LB_odds_df = pd.read_csv('LB_odds.csv')[['LBH', 'LBD', 'LBA']]

    # Load DataFrame with results
    match_results_df = pd.read_csv('match_results.csv')[['result']]

    # Separate features from label
    B365_features = B365_odds_df.values
    BW_features = BW_odds_df.values
    IW_features = IW_odds_df.values
    LB_features = LB_odds_df.values
    labels = match_results_df.values

    # Gather the features of all betting companies
    features_list = [B365_features, BW_features, IW_features, LB_features]

    # 10-fold cross validation
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    K_fold = KFold(n_splits=10, shuffle=True, random_state=6)

    # Iterate for the features of each betting company
    average_train_accuracy_dict = {}  # to store the train accuracy of each betting company
    average_test_accuracy_dict = {}  # to store the test accuracy of each betting company
    current_betting_company = 'B365'  # to print current betting company on console
    for features in features_list:

        current_split = 1  # to print current split on console
        train_accuracy_list = []  # to store the train accuracy of each split
        test_accuracy_list = []  # to store the test accuracy of each split
        for train_index, test_index in K_fold.split(features, labels):
            # Split data
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            current_one_vs_all = 'draw vs home, away'  # to print current 'one vs all' on console
            for i in range(3):  # one vs all (home, draw, away)
                # Train classifier
                weights = train_lms(x_train, y_train, i)

                # Get the most recent updated weights
                weights_final = weights[-1]

                # Evaluate classifier
                train_accuracy = test_lms(x_train, y_train, weights_final, i)
                train_accuracy_list.append(train_accuracy)
                test_accuracy = test_lms(x_test, y_test, weights_final, i)
                test_accuracy_list.append(test_accuracy)

                # Print information of current split on console
                print('---- Split', current_split, ':', current_betting_company, ':', current_one_vs_all, '----')
                print('Train accuracy --> ', train_accuracy)
                print('Test accuracy --> ', test_accuracy)
                print()

                # Update 'current_one_vs_all' variable
                if current_one_vs_all == 'draw vs home, away':
                    current_one_vs_all = 'home vs draw, away'
                elif current_one_vs_all == 'home vs draw, away':
                    current_one_vs_all = 'away vs home, draw'

            current_split += 1  # update variable

        # Print average train and test accuracy on console
        print('-- ', current_betting_company, ' --')
        print('-- Average train accuracy : ', np.average(train_accuracy_list), ' --')
        average_train_accuracy_dict[current_betting_company] = np.average(train_accuracy_list)
        print('-- Average test accuracy : ', np.average(test_accuracy_list), ' --')
        average_test_accuracy_dict[current_betting_company] = np.average(train_accuracy_list)
        print()

        # Update betting company
        if current_betting_company == 'B365':
            current_betting_company = 'BW'
        elif current_betting_company == 'BW':
            current_betting_company = 'IW'
        elif current_betting_company == 'IW':
            current_betting_company = 'LB'

    # Print company is better at training and testing on console
    best_training_company = max(average_train_accuracy_dict, key=average_train_accuracy_dict.get)
    print('Best at training --> ', best_training_company)
    best_testing_company = max(average_test_accuracy_dict, key=average_test_accuracy_dict.get)
    print('Best at training --> ', best_testing_company)
    print()


def train_lms(features, labels, code):
    n = len(features)  # number of samples
    lr = 0.001  # learning rate
    x = features  # input matrix (2 dimensional array)
    desired = one_vs_all(labels, code)  # desired values of each given sample
    errors = []  # history of all errors
    weights_zero = [0, 0, 0]  # array with initial weights
    weights = [weights_zero]  # history of all weights (2 dimensional array)

    for i in range(n):
        current_desired = desired[i]  # desired value of the current sample
        current_x = x[i]  # current sample
        current_weights = weights[i]  # array with current weights

        # Calculate error
        current_error = calculate_error(current_desired, current_x, current_weights)  # current error
        errors.append(current_error)

        # Update coefficients
        new_weights = current_weights + lr * np.dot(current_x, current_error)  # array with updated weights
        weights.append(new_weights)

    return weights


def test_lms(features, labels, w, code):
    n = len(features)  # number of samples
    x = features  # input matrix (2 dimensional array)
    desired = one_vs_all(labels, code)  # desired values of each given sample

    # Compute the output of the classifier (predict labels)
    y = []  # predicted labels
    for i in range(n):
        current_predicted_label = np.dot(x[i].transpose(), w)
        if current_predicted_label < 0:
            current_predicted_label = -1
        elif current_predicted_label > 0:
            current_predicted_label = 1
        y.append(current_predicted_label)

    # Compute accuracy
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    accuracy = accuracy_score(y_true=desired, y_pred=y)

    return accuracy


def one_vs_all(labels, code):
    # The output of the match that is represented by variable 'code' is labeled with value -1
    # The other two outputs are labeled with value 1
    # code = 0 -> draw (D)
    # code = 1 -> home team wins (H)
    # code = 2 -> away team wins (A)

    desired = []
    for value in labels:
        if code == 0:
            if value == 1:  # home team wins
                desired.append(1)
            elif value == 0:  # draw
                desired.append(-1)
            elif value == 2:  # away team wins
                desired.append(1)
        elif code == 1:
            if value == 1:  # home team wins
                desired.append(-1)
            elif value == 0:  # draw
                desired.append(1)
            elif value == 2:  # away team wins
                desired.append(1)
        elif code == 2:
            if value == 1:  # home team wins
                desired.append(1)
            elif value == 0:  # draw
                desired.append(1)
            elif value == 2:  # away team wins
                desired.append(-1)

    return desired


def calculate_error(d, x, w):
    error = d - np.dot(x.transpose(), w)
    return error
