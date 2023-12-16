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

    # Read the data set store in a list and return
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

# select values according to the given state and year values
    for row in dataset_to_use:
        if row[1] == state and row[0].find(year) > - 1:
            dataset_to_use2.append(row)

# extract positive diagnosis values
    for element in dataset_to_use2:
        x = int(element[4])
        return_list.append(x)

    return return_list
    pass


# TODO: Implement this function!
def get_dp_histogram(dataset, state, year, epsilon, N):

    dataset_to_use = dataset.copy()
    dataset_to_use2 = []
    return_list = []

# select values according to the given state and year values
    for row in dataset_to_use:
        if row[1] == state and row[0].find(year) > - 1:
            dataset_to_use2.append(row)

# extract positive diagnosis values by adding the laplace noise
    for element in dataset_to_use2:
        noise = np.random.laplace(0, N/epsilon)
        x = int(element[4])
        x += noise
        return_list.append(x)

    return return_list
    pass


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):

    res = 0

# find the absolute value difference of each bin
    differences = []
    for elem1, elem2 in zip(actual_hist, noisy_hist):
        differences.append(abs(elem1 - elem2))

# take the average of the absolute value differences of each bin
    res = np.sum(differences) / len(actual_hist)
    return res

    pass


# TODO: Implement this function!
def epsilon_experiment(dataset, state, year, eps_values, N):

    res = []

# for each epsilon value repeat 10 times: get actual and noisy histogram calculate avverage error and take average
    for epsilon in eps_values:
        res2 = []
        for i in range(10):
            actual_hist = get_histogram(dataset, state, year)
            noisy_hist = get_dp_histogram(dataset, state, year, epsilon, N)
            res2.append(calculate_average_error(actual_hist, noisy_hist))
        res.append(np.average(res2))

    return res
    pass


# TODO: Implement this function!
def N_experiment(dataset, state, year, epsilon, N_values):

    # same as the epsilon experiment. Here epsilon is fixed and N changes
    res = []
    for n in N_values:
        res2 = []
        for i in range(10):
            actual_hist = get_histogram(dataset, state, year)
            noisy_hist = get_dp_histogram(dataset, state, year, epsilon, n)
            res2.append(calculate_average_error(actual_hist, noisy_hist))
        res.append(np.average(res2))

    return res
    pass


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def max_deaths_exponential(dataset, state, year, epsilon):
    dataset_to_use = dataset.copy()
    dataset_to_use2 = []
    deaths = []
    probabilites = []
    probabilites2 = []

# select values according to the given state and year values
    for row in dataset_to_use:
        if row[1] == state and row[0].find(year) > - 1:
            dataset_to_use2.append(row)

# select death values
    for element in dataset_to_use2:
        x = int(element[2])
        deaths.append(x)

# calculate probabilites of each month to be selected according to the exponential mechanism probablity
    for d in deaths:
        numerator = math.exp((epsilon * d) / 2)
        probabilites.append(numerator)

    denumerator = np.sum(probabilites)

    for p in probabilites:
        probabilites2.append(p / denumerator)

# select and retrun one of the months according to the calculated probabilites
    months = ['01', '02', '03', '04', '05',
              '06', '07', '08', '09', '10', '11', '12']

    x = np.random.choice(months, p=probabilites2)

    return x

    pass


# TODO: Implement this function!
def exponential_experiment(dataset, state, year, epsilon_list):

    res = []
    dataset_to_use = dataset.copy()
    deaths = []
    dataset_to_use2 = []

    for row in dataset_to_use:
        if row[1] == state and row[0].find(year) > - 1:
            dataset_to_use2.append(row)

    for element in dataset_to_use2:
        x = int(element[2])
        deaths.append(x)

    months = ['01', '02', '03', '04', '05',
              '06', '07', '08', '09', '10', '11', '12']

    max_death = np.max(deaths)
    max_death_index = deaths.index(max_death)
    # print("max_death: ", max_death)
    # print("max_death_index: ", max_death_index)

    correct_answer = months[max_death_index]

    for epsilon in epsilon_list:
        count = 0
        for i in range(10000):
            if correct_answer == max_deaths_exponential(dataset, state, year, epsilon):
                count += 1
        res.append(count / 100)

    return res
    pass


# FUNCTIONS TO IMPLEMENT END #


def main():
    filename = "covid19-states-history.csv"
    dataset = read_dataset(filename)

    state = "TX"
    year = "2020"

    # months = ['01', '02', '03', '04', '05',
    #           '06', '07', '08', '09', '10', '11', '12']

    # res = get_histogram(dataset,state,year)
    # plt.bar(months, res)
    # plt.title(f'Positive Test Case for State {state} in year {year}')
    # plt.show()

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
