import numpy as np
from matplotlib import pyplot as plt
# from shapely import geometry, ops

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
          11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

""" Helpers """


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


def plot_grid(cell_percentages):
    max_lat = -8.58
    max_long = 41.18
    min_lat = -8.68
    min_long = 41.14

    background_image = plt.imread('porto.png')

    fig, ax = plt.subplots()
    ax.imshow(background_image, extent=[
              min_lat, max_lat, min_long, max_long], zorder=1)

    rec = [(min_lat, min_long), (min_lat, max_long),
           (max_lat, max_long), (max_lat, min_long)]
    nx, ny = 4, 5  # number of columns and rows  4,5

    polygon = geometry.Polygon(rec)
    minx, miny, maxx, maxy = polygon.bounds
    dx = (maxx - minx) / nx  # width of a small part
    dy = (maxy - miny) / ny  # height of a small part
    horizontal_splitters = [geometry.LineString(
        [(minx, miny + i * dy), (maxx, miny + i * dy)]) for i in range(ny)]
    vertical_splitters = [geometry.LineString(
        [(minx + i * dx, miny), (minx + i * dx, maxy)]) for i in range(nx)]
    splitters = horizontal_splitters + vertical_splitters

    result = polygon
    for splitter in splitters:
        result = geometry.MultiPolygon(ops.split(result, splitter))

    grids = list(result.geoms)

    for grid_index, grid in enumerate(grids):
        x, y = grid.exterior.xy
        ax.plot(x, y, color='#6699cc', alpha=0.7,
                linewidth=3, solid_capstyle='round', zorder=2)

        count = cell_percentages[grid_index]
        count = round(count, 2)

        centroid = grid.centroid
        ax.annotate(str(count) + '%', (centroid.x, centroid.y), color='black', fontsize=12,
                    ha='center', va='center', zorder=3)

    plt.show()


# You can define your own helper functions here. #

### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    probs = []
    d = np.max(DOMAIN)
    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    q = (1 - p) / (d - 1)

    for i in DOMAIN:
        if val == i:
            probs.append(p)
        else:
            probs.append(q)

    res = np.random.choice(DOMAIN, p=probs)
    return res

    pass


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):

    Ivs = np.zeros(len(DOMAIN))
    estimation = []
    d = np.max(DOMAIN)
    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    q = (1 - p) / (np.max(DOMAIN) - 1)
    n = len(perturbed_values)

    for i in perturbed_values:
        Ivs[int(i) - 1] += 1

    for i in Ivs:
        estimation.append((((i - (n*q)) / (p - q))
                           * 100) / len(perturbed_values))

    return estimation
    pass


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):

    n = len(dataset)

    average_error = 0
    actual_values = np.zeros(np.max(DOMAIN))
    perturbed_dataset = []
    perturbed_values = np.zeros(np.max(DOMAIN))

    for i in dataset:
        actual_values[i - 1] += 1
        perturbed_dataset.append(perturb_grr(i, epsilon))

    perturbed_values = estimate_grr(perturbed_dataset, epsilon)
    actual_values = actual_values * (100 / n)

    average_error = np.sum(
        np.abs(perturbed_values - actual_values)) / np.max(DOMAIN)

    return average_error
    pass


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):

    one_hot_encoded = np.zeros(np.max(DOMAIN), dtype=int)
    one_hot_encoded[val - 1] = 1

    return one_hot_encoded
    pass


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):

    p = (np.exp(epsilon / 2)) / (np.exp(epsilon / 2) + 1)
    q = 1 / (np.exp(epsilon / 2) + 1)
    res = []
    probs = []
    zeros = []
    probs.append(p)
    probs.append(q)
    zeros.append(1)
    zeros.append(0)

    for i in encoded_val:
        x = np.random.choice(zeros, p=probs)
        if x == 0:
            if i == 0:
                res.append(1)
            elif i == 1:
                res.append(0)
        elif x == 1:
            res.append(i)

    return res

    pass


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):

    p = (np.exp(epsilon / 2)) / (np.exp(epsilon / 2) + 1)
    q = 1 / (np.exp(epsilon / 2) + 1)
    n = len(perturbed_values)
    index_sums = np.sum(perturbed_values, axis=0)
    estimated_vals = []

    for i in index_sums:
        x = ((i - (n*q)) / (p - q) * 100) / n
        estimated_vals.append(x)

    return estimated_vals

    pass


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):

    average_error = 0
    actual_values = np.zeros(np.max(DOMAIN))
    perturbed_dataset = []
    perturbed_values = np.zeros(np.max(DOMAIN))
    dataset_size = len(dataset)

    for i in dataset:
        actual_values[i - 1] += ((1 * 100) / dataset_size)
        encoded_i = encode_rappor(i)
        perturbed_dataset.append(perturb_rappor(encoded_i, epsilon))

    perturbed_values = estimate_rappor(perturbed_dataset, epsilon)

    average_error = np.sum(
        np.abs(perturbed_values - actual_values)) / np.max(DOMAIN)

    return average_error

    pass


# OUE

# TODO: Implement this function!
def encode_oue(val):

    one_hot_encoded = np.zeros(np.max(DOMAIN), dtype=int)
    one_hot_encoded[val - 1] = 1

    return one_hot_encoded
    pass


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):

    zeros = []
    res = []
    zeros.append(1)
    zeros.append(0)

    for i in encoded_val:
        if i == 1:
            p = []
            p.append(0.5)
            p.append(0.5)
            x = np.random.choice(zeros, p=p)
            res.append(x)
        elif i == 0:
            p = []
            k = 1/(np.exp(epsilon) + 1)
            p.append(k)
            p.append(1 - k)
            x = np.random.choice(zeros, p=p)
            res.append(x)

    return res

    pass


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):

    n = len(perturbed_values)
    index_sums = np.sum(perturbed_values, axis=0)
    estimated_vals = []

    for i in index_sums:
        nominator = 2*(((np.exp(epsilon) + 1) * i) - n)
        denumaretor = np.exp(epsilon) - 1
        estimated_vals.append(((nominator / denumaretor) * 100) / n)

    return estimated_vals

    pass


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):

    average_error = 0
    actual_values = np.zeros(np.max(DOMAIN))
    perturbed_dataset = []
    perturbed_values = np.zeros(np.max(DOMAIN))
    dataset_size = len(dataset)

    for i in dataset:
        actual_values[i - 1] += ((1 * 100) / dataset_size)
        encoded_i = encode_oue(i)
        perturbed_dataset.append(perturb_oue(encoded_i, epsilon))

    

    perturbed_values = estimate_rappor(perturbed_dataset, epsilon)

    average_error = np.sum(
        np.abs(perturbed_values - actual_values)) / np.max(DOMAIN)

    return average_error

    pass


def main():
    dataset = read_dataset("taxi-locations.dat")

    # print("GRR EXPERIMENT")
    # for epsilon in [0.01, 0.1, 0.5, 1, 2]:
    #     error = grr_experiment(dataset, epsilon)
    #     print("e={}, Error: {:.3f}".format(epsilon, error))

    # print("*" * 50)

    # print("RAPPOR EXPERIMENT")
    # for epsilon in [0.01, 0.1, 0.5, 1, 2]:
    #     error = rappor_experiment(dataset, epsilon)
    #     print("e={}, Error: {:.3f}".format(epsilon, error))

    # print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))


if __name__ == "__main__":
    main()
