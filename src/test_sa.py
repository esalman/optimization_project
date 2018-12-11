import csv
import time
from TSP import *

num_of_points = 10
MaxIterations = 100
C = 1e-2
M = 10

list_of_coords = [[4, 0], [2, 2], [5, 3], [5, 6], [1, 6], [4, 9], [9, 5]]

problem_ = TSP(list_of_coords, num_of_points, MaxIterations, C, M)
min_d, fun_evals, probs, dists = problem_.best_tour_SA()

