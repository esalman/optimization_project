from ctf import functions2d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping

matplotlib.rcParams['figure.dpi'] = 100

def minimize_callable(x):
    return fun.cost(x)

# functions
functions = ['Ackley', # Many Local Minima
             'Bukin6',
             'CrossInTray',
             'DropWave',
             'Eggholder',
             'Griewank',
             'HolderTable',
             'Levy13',
             'Rastrigin',
             'Schaffer2',
             'Schaffer4',
             'Schwefel',
             'Shubert', # Bowl Shaped
             'Bohachevsky1',
             'Bohachevsky2',
             'Bohachevsky3',
             'Perm',
             'RotatedHyperEllipsoid',
             'Sphere',
             'SumOfDifferentPowers',
             'SumSquares',
             'Trid', # Plate-Shaped
             'Booth',
             'Matyas',
             'McCormick',
             'PowerSum',
             'Zakharov', # Valley-Shaped
             'ThreeHumpCamel',
             'SixHumpCamel',
             'DixonPrice',
             'Rosenbrock', # Steep Ridges/Drops
             'Absolute',
             'AbsoluteSkewed',
             'DeJong5',
             'Easom',
             'Michalewicz', # Other
             'Beale',
             'Branin',
             'GoldsteinPrice',
             'StyblinskiTang']

fun = functions2d.Rosenbrock()

#for f in functions:
#    target_fun = getattr(functions2d, f)
#    fun = target_fun()
#    
##    fun.domain = np.array([[-5, 5], [-5, 5]])
#    fun.plot_cost(points=100)
#    plt.show()

x0 = np.array([-0.3, 10])
print('\nBFGS\n')
err = np.array([])
res = minimize(minimize_callable, x0, method='BFGS', tol=1e-6)
print(res)

res = basinhopping(minimize_callable, x0, minimizer_kwargs={"method": "BFGS"})
print('\nbasinhopping\n', res)
print('global min', fun.min)


