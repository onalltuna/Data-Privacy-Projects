##############################################################################
# This skeleton was created by Efehan Guner (efehanguner21@ku.edu.tr)    #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
import glob
import os
import sys
from copy import deepcopy
import time
from typing import Optional
import numpy as np

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################


# Tree structure is defined to used for storing DGHs and to be used in Top-down anonymizer
class Tree:

    def __init__(self, data) -> None:
        self.data = data
        self.children = []

    def add_child(self, child) -> None:
        self.children.append(child)


# This function is used to store a DGH in a tree. It takes the elemets of the DGH and the number of "tab" characters
# that corresponds to each element and their location inside the tree.
# This function enables DGHs to be stored in trees and to be used in various operations
def construct_tree(elements, ts):
    tree = Tree(elements[0])
    current = tree
    current_level = 0
    stack = [tree]
    x = 1

    # for e in elements[1:]:
    #     new_node = Tree(e)
    for t in ts[1:]:
        new_node = Tree(elements[x])
        if t > current_level:
            current.add_child(new_node)
            stack.append(new_node)
            x += 1
            current_level = t
            current = new_node
        else:
            while ts[elements.index(stack[-1].data)] >= t:
                stack.pop()
            stack[-1].add_child(new_node)
            stack.append(new_node)
            current = new_node
            current_level = t
            x += 1

    return tree


# This funcion is used to find the path in a tree starting from the given node to the target value
def find_path(node, target, path):

    path.append(node.data)

    if node.data == target:
        return path

    for child in node.children:
        new_path = find_path(child, target, path.copy())
        if new_path:
            return new_path

    return []


# This function is used to return the node which holds the target value
# Sometimes on DGHs we have the value string but we need it as a node to perform some actions
def find_child(tree, target):

    if tree.data == target:
        return tree

    for child in tree.children:
        new_child = find_child(child, target)
        if new_child:
            return new_child


# This function is used to find the number of descendent leaf nodes of a given node it is used in LM_cost calculations
def find_num_descendent(node, count):

    if node.children == None or len(node.children) == 0:
        return 1

    for child in node.children:
        count += find_num_descendent(child, 0)

    return count


# This function is used to find the most common_ancestor of a list of elements in a given tree
# Tree here is a DGH and list is one of the attributes of all the elements in a cluster to be anonymized
# This function is used when only minimum amount of anonymization is needed
def common_ancestor(tree, list):
    stack = []
    paths = []

    for i in list:
        paths.append(find_path(tree, i, stack))
        stack = []

    common = paths[0][0]
    x = 1

    for i in range(min(map(len, paths))):
        new_common = paths[0][i]
        for p in paths:
            if p[i] != new_common:
                x = 0
        if x == 1:
            common = new_common

    return common


# This function is used to apply minimum amount of anonymization to the given cluster according to the given DHGs
# It finds the most common ancestor of the cluster for each QI. Then equalizes each QI to the common ancestor
def anonymize_minimum(cluster, dhgs):

    x = 0
    keys = []
    keys = list(cluster[0].keys())

    values = []
    commons = []

    for k in keys:
        values2 = []
        values.append(values2)
        for i in cluster:
            values[x].append(i.get(k))
        x += 1

    for i in range(len(dhgs)):
        commons.append(common_ancestor(dhgs.get(keys[i]), values[i]))

    for i in cluster:
        for j in range(len(commons)):
            i[keys[j]] = commons[j]

    return cluster


def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    # print(result[0]['age']) # debug: testing.
    # print(result)
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset) > 0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True


def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    # TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.

    list2 = []
    t_counts = []
    t_num1 = 0
    # count the number of "tab" characters corresponding to the each element of the DGH to be used in construct_tree fucntion
    with open(DGH_file, 'r') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            text = ' '.join(line)
            # list.append(row)
            # print(text.strip())
            while text.startswith('\t'):
                t_num1 += 1
                text = text[1:]
            list2.append(text)
            t_counts.append(t_num1)
            t_num1 = 0

        dgh = construct_tree(list2, t_counts)
        # print(list)
        # print(list2)
        # print(t_counts)
        return dgh
        pass


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


def key_to_specialize(node, DGHs, k):

    keys = list(node.data.keys())
    keys.remove("records")
    values = []
    counts_dictionary = {}
    all_children = {}

    # get the QI values of the given node from the top-down anonymization tree
    for key in keys:
        values.append(node.data[key])

    for key in keys:
        counts_dictionary.update({key: []})

    # find the number of possible splits by choosing each QI value
    for value, key in zip(values, keys):
        current = find_child(DGHs[key], value)
        all_children.update({key: current.children})

    for attribute in all_children:
        child_attributes = all_children[attribute]
        len_child = len(child_attributes)
        if len_child == 0:
            counts_dictionary[attribute].append(0)
        for i in range(len_child):
            counts_dictionary[attribute].append(0)

    for attribute in all_children:
        child_attributes = all_children[attribute]
        len_child = len(child_attributes)
        for record in node.data["records"]:
            path = []
            path2 = find_path(DGHs[attribute], record[attribute], path)
            x = 0
            for c in child_attributes:
                if c.data in path2:
                    counts_dictionary[attribute][x] += 1
                x += 1

    # remove the QI values which vioaltes the k anonymity
    counts_dictionary2 = counts_dictionary.copy()
    for attribute in counts_dictionary:
        mini = min(counts_dictionary[attribute])
        if mini < k:
            del counts_dictionary2[attribute]
            if len(counts_dictionary2) == 0:
                return counts_dictionary2

    keys2 = list(counts_dictionary2.keys())
    min_len = len(counts_dictionary2[keys2[0]])

    # among those who satisfy k anonymity find the possible number of splits
    for attribute in counts_dictionary2:
        if len(counts_dictionary2[attribute]) < min_len:
            min_len = len(counts_dictionary2[attribute])

    # remove the QI values which leads to higher splits than the minimum possible split
    counts_dictionary3 = counts_dictionary2.copy()
    for attribute in counts_dictionary2:
        if len(counts_dictionary2[attribute]) > min_len:
            del counts_dictionary3[attribute]

    keys2 = list(counts_dictionary3.keys())
    min_distance = calculate_distance(counts_dictionary3[keys2[0]])
    prev_attribute = keys2[0]

    counts_dictionary4 = counts_dictionary3.copy()

    # If there are still more than one possible QI values
    # select the QI values that has the minimum distance according to the L1 norm
    for attribute in counts_dictionary3:
        current_distance = calculate_distance(counts_dictionary4[attribute])
        if current_distance < min_distance:
            min_distance = current_distance
            del counts_dictionary4[prev_attribute]
            prev_attribute = attribute
        elif current_distance > min_distance:
            del counts_dictionary4[attribute]

    return counts_dictionary4


# This function is used to calculate distance to be used in top-down anonymizer
# It makes tha calculation according to the L1 norm from project description
def calculate_distance(numbers):
    num_len = len(numbers)
    x = 1 / num_len
    distance = 0
    summed = sum(numbers)

    for num in numbers:
        distance += abs(x - (num / summed))

    # print("\ndistance in calculate: ", distance)

    return distance


# This function returns the leaves of the tree with given root node
# It is used in top-down anonymizer to find leaf nodes to apply specialization
def find_leaves(root):
    leaf_nodes = []

    def iterate(node):
        if node.children == None or len(node.children) == 0:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                iterate(child)

    iterate(root)
    return leaf_nodes


def specialize(leaves, DGHs, k):

    # apply specialization on each leaf
    for leaf in leaves:

        # find which DGH to use in specialization
        key1 = list(key_to_specialize(leaf, DGHs, k).keys())

        # if there is a DGH that is suitable to use in specialization find the children of the current QI value
        if key1 is not None and len(key1) != 0:
            key = key1[0]
            leaf_data = leaf.data
            key_value = leaf_data[key]
            key_node = find_child(DGHs[key], key_value)

            # for each children construct a new tree node and append it to the children of the current leaf
            for child in key_node.children:

                new_dict = leaf.data.copy()
                new_dict[key] = child.data
                new_dict["records"] = []
                for record in leaf.data["records"]:
                    val_to_change = record[key]
                    path = []
                    path2 = find_path(DGHs[key], val_to_change, path)
                    if child.data in path2:
                        new_dict["records"].append(record)
                new_tree_node = Tree(new_dict)
                leaf.add_child(new_tree_node)


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    # TODO: complete this function.

#  To find MD cost find the length of the path(starting from the QI in anonymized dataset to the QI in raw dataset) in respective DGH
    total_cost = 0
    for row1, row2 in zip(anonymized_dataset, raw_dataset):
        for i in DGHs:
            path = []
            path2 = find_path(find_child(DGHs[i], row1[i]), row2[i], path)
            total_cost += (len(path2) - 1)

    return total_cost


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    # TODO: complete this function.

    M = len(DGHs)
    W = 1 / M

    LM_val = 0
    LM_rec = 0
    LM_anon = 0

# To calculate LM(val) find the number of descendant leaves of val - 1  divide it to the total number of leaves in the DGH - 1
# For a record sum over all the values and for the dataset sum over all the records
    for r in anonymized_dataset:
        LM_rec = 0
        for i in DGHs:
            y = find_child(DGHs[i], r[i])
            nominator = find_num_descendent(y, 0) - 1
            denominator = find_num_descendent(DGHs[i], 0) - 1
            LM_val = nominator / denominator
            LM_val *= W
            LM_rec += LM_val

        LM_anon += LM_rec

    return LM_anon


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                      output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    for i in range(len(raw_dataset)):  # set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s)  # to ensure consistency between runs
    np.random.shuffle(raw_dataset)  # shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)

    # TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters".
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    num_cluster = int(D / k)
    x = 1
    y = 0
    cluster = []

# First divide all records to the clusters x holds the number of records in the cluster y holds the number of clusters.
# If dataset size is not perfect multiple of k last cluster has more records than others
    for i in raw_dataset:
        if y == num_cluster-1:
            cluster.append(i)

        elif x != k:
            cluster.append(i)
            x += 1
        elif x == k:
            cluster.append(i)
            clusters.append(cluster)
            cluster = []
            x = 1
            y += 1

    clusters.append(cluster)

# Then anonymize each cluster by using anonymize_minumum funtion that implements minumum amount of anonymization
    for cluster in clusters:
        cluster = anonymize_minimum(cluster, DGHs)

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:  # restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                          output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    # TODO: complete this function.
    M = len(DGHs)
    W = 1 / M
    D = len(raw_dataset)
    dictionary = {}
    clusters = []

    for i in range(len(raw_dataset)):  # set indexing to not lose original places of records
        raw_dataset[i]['index'] = i
    raw_dataset = np.array(raw_dataset)

# At first all the raw_dataset is unused
    unused = raw_dataset

# While there are at least k unused records keep going
    while len(unused) >= k:
        # Take the first value from unused and delete it from unused
        rec = unused[0]
        unused = np.delete(unused, 0)

        # For the rest of the records in unused, construct a mini cluster
        # with 2 records(the first record previously deleted from unused + next record in unused)
        # then calculate the LM cost of this mini cluster and store it as the LM cost of each record in unused
        for row in unused:
            row2 = row.copy()
            rec2 = rec.copy()
            mini_cluster = []
            mini_cluster.append(rec2)
            mini_cluster.append(row2)
            mini_cluster2 = anonymize_minimum(mini_cluster, DGHs)

            LM_anon = 0
            for r in mini_cluster2:
                LM_rec = 0
                for i in DGHs:
                    y = find_child(DGHs[i], r[i])
                    nominator = find_num_descendent(y, 0) - 1
                    denominator = find_num_descendent(DGHs[i], 0) - 1
                    LM_val = nominator / denominator
                    LM_val *= W
                    LM_rec += LM_val
                LM_anon += LM_rec

            # Store LM cost of each raw in a dictionary which has index as the key
            info = []
            info.append(LM_anon)
            info.append(row)
            dictionary.update({row["index"]: info})

        # Sort this dictionary according to the their LM costs in increasing order and take the first k-1 elements
        sorted_dic = dict(
            sorted(dictionary.items(), key=lambda item: item[1][0]))
        first_n_elements = {key: sorted_dic[key]

                            for key in list(sorted_dic)[:k-1]}
        # Append each record from first k-1 list to a cluster and remove them from the unused list
        # Anonymize the cluster and append it to the clusters
        cluster = []
        cluster.append(rec)
        for key, value in first_n_elements.items():
            cluster.append(value[1])
            filtered_unused = [item for item in unused if item['index'] != key]
            unused = filtered_unused
        dictionary = {}
        cluster2 = anonymize_minimum(cluster, DGHs)
        clusters.append(cluster2)

    # If there are any more records in unused add them to the last cluster and anonymize the last cluster
    j = len(unused)
    if j > 0:
        for r in unused:
            cluster.append(r)
        cluster2 = anonymize_minimum(cluster, DGHs)

    # Reorder the records according to ther indexes index value is stored in the dictionary for this operation
    anonymized_dataset = [None] * D

    for d in clusters:  # restructure according to previous indexes
        for item in d:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


def topdown_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                       output_file: str):
    """ Top-down anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    # TODO: complete this function.
    D = len(raw_dataset)
    dictionary = {}
    clusters = []

    for i in range(len(raw_dataset)):  # set indexing to not lose original places of records
        raw_dataset[i]['index'] = i
    raw_dataset = np.array(raw_dataset)

    keys = DGHs.keys()

    for key in keys:
        dictionary.update({key: DGHs[key].data})

    dictionary.update({"records": raw_dataset})

    # A dictionary is used to represent each node of the top-down anonymizer tree
    # Keys of this dictionary are the DGHs and one key for the records
    # Value of the records key is all the records that are included in the cluster that is decided by the DGH values
    tree = Tree(dictionary)
    leaves = find_leaves(tree)
    leaves2 = []

    # leaves is the current leaves of the top-down anonymizer tree
    # apply specialization on each leaf
    # if there is no change in leaves maximum specialization is achieved
    while leaves != leaves2:
        leaves = find_leaves(tree)
        specialize(leaves, DGHs, k)
        leaves2 = find_leaves(tree)

    # take the records from each leaf and store them in clusters to be able to restructure and write to result dataset
    for leaf in find_leaves(tree):
        for key in keys:
            for record in leaf.data["records"]:
                record[key] = leaf.data[key]
        clusters.append(leaf.data["records"])

    anonymized_dataset = [None] * D

    for d in clusters:  # restructure according to previous indexes
        for item in d:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(
        f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
    print(f"\tWhere algorithm is one of [clustering, random, topdown]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'topdown']:
    print("Invalid algorithm.")
    sys.exit(2)

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

# start time to measure time cost of algorithms
start_time = time.perf_counter()

function = eval(f"{algorithm}_anonymizer")
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(
            f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, topdown]")
        sys.exit(1)

    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)

# start time to measure time cost of algorithms
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

print(
    f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")


# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300
