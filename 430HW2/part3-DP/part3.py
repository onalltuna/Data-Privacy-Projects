import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random
import csv

from numpy import sqrt, exp


''' Functions to implement '''

# TODO: Implement this function!


def read_dataset(file_path):

    data_set = []
    data_set_row = []

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            for element in row:
                data_set_row.append(element)
            data_set_row_copy = data_set_row.copy()
            data_set.append(data_set_row_copy)
            data_set_row.clear()

    # print(data_set)
    return data_set
    pass


# TODO: Implement this function!
def get_histogram(dataset, state='TX', year='2020'):

    dataset_to_use = dataset.copy()
    dataset_to_use2 = []
    return_list = []
    print("state: ", state)

    for row in dataset_to_use:
        if row[1] == state and row[0].find(year) > - 1:
            dataset_to_use2.append(row)

    for element in dataset_to_use2:
        return_list.append(element[4])

    # print(dataset_to_use2)
    print("return_list: ", return_list)

    months = ['01', '02', '03', '04', '05',
              '06', '07', '08', '09', '10', '11', '12']

    plt.bar(months, return_list)
    plt.title(f'Positive Test Case for State {state} in year {year}')
    plt.show()

    return return_list
    pass


# TODO: Implement this function!
def get_dp_histogram(dataset, state, year, epsilon, N):
    pass


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    pass


# TODO: Implement this function!
def epsilon_experiment(dataset, state, year, eps_values, N):
    pass


# TODO: Implement this function!
def N_experiment(dataset, state, year, epsilon, N_values):
    pass


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def max_deaths_exponential(dataset, state, year, epsilon):
    pass


# TODO: Implement this function!
def exponential_experiment(dataset, state, year, epsilon_list):
    pass


# FUNCTIONS TO IMPLEMENT END #


def main():
    filename = "covid19-states-history.csv"
    dataset = read_dataset(filename)

    h = get_histogram(dataset)

    state = "TX"
    year = "2020"

    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg = epsilon_experiment(dataset, state, year, eps_values, 2)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])

    print("**** N EXPERIMENT RESULTS ****")
    N_values = [1, 2, 4, 8]
    error_avg = N_experiment(dataset, state, year, 0.5, N_values)
    for i in range(len(N_values)):
        print("N = ", N_values[i], " error = ", error_avg[i])

    state = "WY"
    year = "2020"

    print("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]
    exponential_experiment_result = exponential_experiment(
        dataset, state, year, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ",
              exponential_experiment_result[i])


if __name__ == "__main__":
    main()
