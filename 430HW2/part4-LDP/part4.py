import numpy as np
from matplotlib import pyplot as plt
from shapely import geometry, ops

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

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
    ax.imshow(background_image, extent=[min_lat, max_lat, min_long, max_long], zorder=1)

    rec = [(min_lat, min_long), (min_lat, max_long), (max_lat, max_long), (max_lat, min_long)]
    nx, ny = 4, 5  # number of columns and rows  4,5

    polygon = geometry.Polygon(rec)
    minx, miny, maxx, maxy = polygon.bounds
    dx = (maxx - minx) / nx  # width of a small part
    dy = (maxy - miny) / ny  # height of a small part
    horizontal_splitters = [geometry.LineString([(minx, miny + i * dy), (maxx, miny + i * dy)]) for i in range(ny)]
    vertical_splitters = [geometry.LineString([(minx + i * dx, miny), (minx + i * dx, maxy)]) for i in range(nx)]
    splitters = horizontal_splitters + vertical_splitters

    result = polygon
    for splitter in splitters:
        result = geometry.MultiPolygon(ops.split(result, splitter))

    grids = list(result.geoms)

    for grid_index, grid in enumerate(grids):
        x, y = grid.exterior.xy
        ax.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

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
    pass


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    pass


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    pass


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    pass


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    pass


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    pass


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    pass


# OUE

# TODO: Implement this function!
def encode_oue(val):
    pass


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    pass


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    pass


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    pass


def main():
    dataset = read_dataset("taxi-locations.dat")

    print("GRR EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))

    print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))

    print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))
            
    

if __name__ == "__main__":
    main()
