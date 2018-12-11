import numpy as np
import cv2
import math
import pickle
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import itertools
import decimal

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

class TSP:   
    def __init__(self, list_of_coords, num_of_points, MaxIterations, C, M):
        """ Initializes the TSP problem with a list of
        coordinate points. The coordinate points represent
        the locations of each city.
            Example:
                # City coordinates:
                #  City 1: [0, 0]
                #  City 2: [1, 2]
                #  City 3: [5, 3]
                list_of_coords = [[0, 0], [10, 20], [50, 30], [40, 60]]
                
                num_of_points = 100
                MaxIterations = 1000
                C = 1e-2
                M = 0

                my_TSP = TSP(list_of_coords, num_of_points, MaxIterations, C, M)
        """
        
        self.city_coords   = list_of_coords
        self.num_of_cities = len(list_of_coords)
        self.num_of_points = num_of_points
        self.MaxIterations = MaxIterations
        self.C = C
        self.M = M
       
    # Describe the class for debugging. Help can be provided using:
    #    print(repr(<instanceName>))
    def __repr__(self):
        """ Provides information for developers. """
        
        info = """ 
          The TSP class provides tour functions for the TSP
          problem. It computes tour lengths and prints tours.
          """          
        return(info)
        
    def __str__(self):
        """ Provides information for users of the class.
        The information is available using:
            print(name_of_the_object)
        """
        
        info = ("TSP: Number of cities={}\n".format(self.num_of_cities))
        info += ("List of coordinate points:\n")
        info += str(self.city_coords)
        return(info)
    
    # TODO: ???? Create valid_tour function
    
    def tour_length(self, tour):
        """
        Computes the length of a tour.
        The parameter tour lists the city order.
        Example:
            city_coords = [[0, 0], [1, 2], [5, 3]]
            my_TSP    = TSP(city_coords)
            tour1     = [1, 3, 2, 1]
            tour1_len = my_TSP.tour_length(tour1)
            print("tour=", tour1, "has length ", tour1_len)
        """
        
        if (len(tour)-1 != self.num_of_cities):
            print("ERROR in tour_length()")
            print("Number of cities = ", self.num_of_cities)
            print("Tour = ", tour)
            print("does not have enough cities!")
            return
        
        city1 = tour[0]
        if (city1 != 1):
            print("ERROR in tour_length()")
            print("Start city must be 1")
            print("Tour = ", tour)
            print("does not start at 1")
            return
        
        cityN = tour[-1]
        if (cityN != 1):
            print("ERROR in tour_length()")
            print("Final city must be 1")
            print("Tour = ", tour)
            print("does not end at 1")
            return
        
        dist = 0.0    
        for i in range(0, len(tour)-1, 1):
            j1 = tour[i]-1
            j2 = tour[i+1]-1            
            c1 = self.city_coords[j1]
            c2 = self.city_coords[j2]
            hyp = math.sqrt((c1[0]-c2[0])**2 
                            + (c1[1]-c2[1])**2)
            dist += hyp
        
        return(dist)
        
    def plot_tour(self, tour, num_of_rows, num_of_cols):
        """
        Plots a given tour.
        Example:
            list_of_coords = [[0, 0], [10, 20], [50, 30]]
            my_TSP    = TSP(city_coords)
            tour1     = [1, 3, 2, 1]
            tour1_len = my_TSP.tour_length(tour1)
            print("tour=", tour1, "has length ", tour1_len)
            my_TSP.plot_tour(tour1, 100, 100)
        """
        
        # Find the limits:
        c = np.array(self.city_coords)
        row_min = np.min(c[:,0])
        row_max = np.max(c[:,0])
        col_min = np.min(c[:,1])
        col_max = np.max(c[:,1])
                
        # Adjust the text for cities to be reasonable:
        offset = np.min([0.1*(col_max-col_min), 1])
        
        # Plot all of the points.
        plt.figure()
        plt.axis('image')
        for i in range(0, len(tour)-1, 1):            
            j1 = tour[i]-1
            j2 = tour[i+1]-1
            c1 = tuple(self.city_coords[j1])
            c2 = tuple(self.city_coords[j2])
            c1 = (c1[1], c1[0])
            c2 = (c2[1], c2[0])                        
            plt.plot((c1[0], c2[0]), (c1[1], c2[1]), 'b-')
            circle1=plt.Circle(c1, radius=1, color='r', fill=False)
            circle2=plt.Circle(c2, radius=1, color='r', fill=False)                      
            plt.gcf().gca().add_artist(circle1)
            plt.gcf().gca().add_artist(circle2)
            plt.text(c1[0]+offset, c1[1]+offset, str(j1+1), fontsize=14)
            plt.text(c2[0]+offset, c2[1]+offset, str(j2+1), fontsize=14)
            
        # Reverse the limits:
#        plt.ylim(1.2*float(row_max), 0.8*float(row_min))
#        plt.xlim(0.8*float(col_min), 1.2*float(col_max))
        plt.ylim(float(row_max) + 0.1*float(abs(row_max)), float(row_min) - 0.1*float(abs(row_min)))
        plt.xlim(float(col_min) - 0.1*float(abs(col_min)), float(col_max) + 0.1*float(abs(col_max)))
        plt.show()
        
        
    def best_tour_SA(self, plot=False):
        """ Computes the best tour.
        Example:
            city_coords = [[0, 0], [10, 20], [50, 30], [40, 60]]            
            my_TSP = TSP(city_coords)
            my_TSP.best_tour_SA()
        """
        
        min_tours = []
        min_distances = np.zeros(self.num_of_points)
        fun_evals = np.zeros(self.num_of_points)
        probs = []
        dists = []
        
        for i in range(self.num_of_points):
            # generate a random tour
            t = np.arange(2, self.num_of_cities+1)
            t = np.random.permutation(t)
            t = np.insert(t, 0, 1)
            t = np.append(t, 1)
            
            # run simulated annealing
            t_, d_, p_, min_t, min_d, evals = SA(self.tour_length, t, self.C, self.MaxIterations, self.M)
            
#            print( min_t, min_d )
            
            min_tours.append( min_t )
            min_distances[i] = min_d
            fun_evals[i] = evals
            probs.append(p_)
            dists.append(d_)
        
        # Fix rows and cols later
        min_d = np.min( min_distances )
        min_idx = np.argmin( min_distances )
        best_tour = min_tours[min_idx]
        
        if plot:
            self.plot_tour(best_tour, 180, 360)
            print("Best tour using simulated annealing is")
            print(np.ndarray.tolist( best_tour ))
            print("It has distance = ", min_d)
        
        return min_d, fun_evals, probs, dists
        
        
    def best_tour(self):
        """ Computes the best tour.
        Example:
            city_coords = [[0, 0], [10, 20], [50, 30], [40, 60]]            
            my_TSP = TSP(city_coords)
            my_TSP.best_tour()
        """
        
        # Generate all permutations
        basic_list = list(range(2, self.num_of_cities+1, 1))
        all_perms  = list(itertools.permutations(basic_list))
        
        # Generate valid tours:
        i = 1
        min_dist = math.inf
        for tour in all_perms:
            t = list(tour)
            t = [1] + t
            t.append(1)
                        
            dist = self.tour_length(t)
            
#            print("Tour = ", t)
#            print("Distance = ", dist)
            
            if (dist < min_dist):
                best_tour = t
                min_dist  = dist
        
        # Fix rows and cols later
        self.plot_tour(best_tour, 100, 100)
        print("Best tour using grid search is")
        print(best_tour)
        print("It has distance = ", min_dist)


def SA(fun, x0, C, MaxIterations, M):
    # initialize
    x = x0
    dx = -fun(x)
    
    # for storing results
    x_ = [x0]
    d_ = np.array(dx)
    p_ = np.array(0)
    
    num_cities = len(x0)-1
    
    for i in range(MaxIterations):
        # get a random neighbor
        j = np.random.choice(np.arange(2, num_cities), size=2, replace=False) # randomly choose 2 cities
        y = np.copy(x)
        y[ j[0] ], y[ j[1] ] = y[ j[1] ], y[ j[0] ] # swap
        
        # new neighbor distance evaluation
        dy = -fun(y)
        
        # evaluate the jump probability
        p = min( 1, decimal.Decimal(2+i)**( decimal.Decimal(C) * decimal.Decimal(dy - dx) ) )
        
#        print(dx, dy, p)
        
        if np.random.uniform(0,1) < p:
            x = y
            dx = dy
        
        x_.append( x )
        d_ = np.append( d_, dx )
        p_ = np.append( p_, p )
        
        # stopping criteria
        if M != 0 and i > M:
            # check if there was a local minima in previous M iterations
            if not np.any( p_[-M:] == 1):
                break
    
    min_d = abs( np.max( d_ ) )
    min_idx = np.argmax( d_ )
    min_x = x_[min_idx]
    
    d_ = abs(d_)

    return x_, d_, p_, min_x, min_d, i