import sys
import random

import numpy as np
import pandas as pd
import copy


from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


###############################################################################
############################# Label Flipping ##################################
###############################################################################
def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p):
    """
    Performs a label flipping attack on the training data.

    Parameters:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    model_type: Type of model ('DT', 'LR', 'SVC')
    p: Proportion of labels to flip

    Returns:
    Accuracy of the model trained on the modified dataset
    """
    # TODO: You need to implement this function!
    # Implementation of label flipping attack

    res = 0
    N = np.shape(X_train)[0]
    num_to_change = int(N * p)


    # print("N: ", N)
    # print("numtochange: ", num_to_change)
    # print("y_train: ", y_train)

    if model_type == "DT":
        for x in range(100):
            indexes_to_change = random.sample(range(0, N - 1), num_to_change)
            y_train3 = y_train.copy()
            for i in indexes_to_change:
                y_train3[i] = 1 - y_train3[i]
            myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
            myDEC.fit(X_train, y_train3)
            DEC_predict = myDEC.predict(X_test)
            accuracy = accuracy_score(y_test, DEC_predict)
            res += accuracy / 100

    elif model_type == "LR":
        for x in range(100):
            indexes_to_change = random.sample(range(0, N - 1), num_to_change)
            y_train3 = y_train.copy()
            for i in indexes_to_change:
                y_train3[i] = 1 - y_train3[i]
            myLR = LogisticRegression(
                penalty='l2', tol=0.001, C=0.1, max_iter=100)
            myLR.fit(X_train, y_train3)
            LR_predict = myLR.predict(X_test)
            accuracy = accuracy_score(y_test, LR_predict)
            res += accuracy / 100

    elif model_type == "SVC":
        for x in range(100):
            indexes_to_change = random.sample(range(0, N - 1), num_to_change)
            y_train3 = y_train.copy()
            for i in indexes_to_change:
                y_train3[i] = 1 - y_train3[i]
            mySVC = SVC(C=0.5, kernel='poly', random_state=0)
            mySVC.fit(X_train, y_train3)
            SVC_predict = mySVC.predict(X_test)
            accuracy = accuracy_score(y_test, SVC_predict)
            res += accuracy / 100

        # print('Accuracy of decision tree: ' +
        #   str(accuracy_score(y_test, DEC_predict)))

    return res
    return -999


###############################################################################
########################### Label Flipping Defense ############################
###############################################################################

def label_flipping_defense(X_train, y_train, p):
    """
    Performs a label flipping attack, applies outlier detection, and evaluates the effectiveness of outlier detection.

    Parameters:
    X_train: Training features
    y_train: Training labels
    p: Proportion of labels to flip

    Prints:
    A message indicating how many of the flipped data points were detected as outliers
    """
    # TODO: You need to implement this function!
    # Perform the attack, then the defense, then print the outcome
    print(f"Out of {0} flipped data points, {0} were correctly identified.")


###############################################################################
############################# Evasion Attack ##################################
###############################################################################
def evade_model(trained_model, actual_example):
    """
    Attempts to create an adversarial example that evades detection.

    Parameters:
    trained_model: The machine learning model to evade
    actual_example: The original example to be modified

    Returns:
    modified_example: An example crafted to evade the trained model
    """
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    # while pred_class == actual_class:
    # do something to modify the instance
    #    print("do something")

    return modified_example


def calc_perturbation(actual_example, adversarial_example):
    """
    Calculates the perturbation added to the original example.

    Parameters:
    actual_example: The original example
    adversarial_example: The modified (adversarial) example

    Returns:
    The average perturbation across all features
    """
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


###############################################################################
########################## Transferability ####################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    """
    Evaluates the transferability of adversarial examples.

    Parameters:
    DTmodel: Decision Tree model
    LRmodel: Logistic Regression model
    SVCmodel: Support Vector Classifier model
    actual_examples: Examples to test for transferability

    Returns:
    Transferability metrics or outcomes
    """
    # TODO: You need to implement this function!
    # Implementation of transferability evaluation

    print("Out of 40 adversarial examples crafted to evade DT :")
    print(f"-> {0} of them transfer to LR.")
    print(f"-> {0} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade LR :")
    print(f"-> {0} of them transfer to DT.")
    print(f"-> {0} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade SVC :")
    print(f"-> {0} of them transfer to DT.")
    print(f"-> {0} of them transfer to LR.")


###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ##
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##
def main():
    data_filename = "BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0)

    # Raw model accuracies:
    print("#" * 50)
    print("Raw model accuracies:")

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' +
          str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' +
          str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    print("#"*50)
    print("Label flipping attack executions:")
    model_types = ["DT", "LR", "SVC"]
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for p in p_vals:
            acc = attack_label_flipping(
                X_train, X_test, y_train, y_test, model_type, p)
            print("Accuracy of poisoned", model_type, str(p), ":", acc)

    # # Label flipping defense executions:
    # print("#" * 50)
    # print("Label flipping defense executions:")
    # p_vals = [0.05, 0.10, 0.20, 0.40]
    # for p in p_vals:
    #     print("Results with p=", str(p), ":")
    #     label_flipping_defense(X_train, y_train, p)

    # # Evasion attack executions:
    # print("#"*50)
    # print("Evasion attack executions:")
    # trained_models = [myDEC, myLR, mySVC]
    # model_types = ["DT", "LR", "SVC"]
    # num_examples = 40
    # for a, trained_model in enumerate(trained_models):
    #     total_perturb = 0.0
    #     for i in range(num_examples):
    #         actual_example = X_test[i]
    #         adversarial_example = evade_model(trained_model, actual_example)
    #         if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
    #             print("Evasion attack not successful! Check function: evade_model.")
    #         perturbation_amount = calc_perturbation(
    #             actual_example, adversarial_example)
    #         total_perturb = total_perturb + perturbation_amount
    #     print("Avg perturbation for evasion attack using",
    #           model_types[a], ":", total_perturb / num_examples)

    # # Transferability of evasion attacks:
    # print("#"*50)
    # print("Transferability of evasion attacks:")
    # trained_models = [myDEC, myLR, mySVC]
    # num_examples = 40
    # evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])


if __name__ == "__main__":
    main()
