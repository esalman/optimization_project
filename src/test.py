from ctf import functions2d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping
from matplotlib.patches import Polygon

matplotlib.rcParams['figure.dpi'] = 100

def cost_callable(x):
    return fun.cost(x)

def jac_callable(x):
    return fun.grad(x)

def log_callback(*args):
    global err, fun, xk, accept
    xk.append( args[0] )
    err = np.append( err, np.linalg.norm(args[0]-fun.min) )
    if len(args) == 3:
        accept = np.append(accept, args[2])
        if accept.shape[0] > 10 and np.sum( accept[-10:] ) == 10:
            return True

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

## Domain Correction
# Lower x0 Limit
if np.isfinite(fun.domain[0][0]):
    x0_lim_lower = fun.domain[0][0]
else:
    x0_lim_lower = -10.0
# Upper x0 Limit
if np.isfinite(fun.domain[0][1]):
    x0_lim_upper = fun.domain[0][1]
else:
    x0_lim_upper = +10.0
# Lower x1 Limit
if np.isfinite(fun.domain[1][0]):
    x1_lim_lower = fun.domain[1][0]
else:
    x1_lim_lower = -10.0
# Upper x1 Limit
if np.isfinite(fun.domain[1][1]):
    x1_lim_upper = fun.domain[1][1]
else:
    x1_lim_upper = +10.0
x0 = np.array([np.random.uniform(x0_lim_lower, x0_lim_upper, 1)[0],
               np.random.uniform(x1_lim_lower, x1_lim_upper, 1)[0]])
print('x0=', x0)

xk = [x0]
err = np.array([])
res = minimize(cost_callable, x0, method='BFGS', jac=jac_callable, tol=1e-6, callback=log_callback)
minimize_err = np.copy(err)
minimize_xk = np.copy(xk)
plt.plot(minimize_err)
print('\nminimize\n', res)
#plt.grid()
#plt.show()

xk = [x0]
err = np.array([])
accept = np.array([])
res = basinhopping(cost_callable, x0, callback=log_callback, niter=100,
                   minimizer_kwargs={'method': 'BFGS', 'jac': jac_callable, 'tol': 1e-6})
basin_err= np.copy(err)
basin_xk = np.copy(xk)
plt.plot(basin_err)
print('\nbasinhopping\n', res)
print('global min', fun.min)

plt.grid()
plt.legend(['minimize', 'basinhopping'])
plt.show()

fun.plot_cost(points=100)

#plt.plot(minimize_xk[:,0], minimize_xk[:,1], '.-g', markersize=3, linewidth=1)
#plt.plot(minimize_xk[-1,0], minimize_xk[-1,1], 'xk')

plt.plot(basin_xk[:,0], basin_xk[:,1], '.-b', markersize=3, linewidth=1)
plt.plot(basin_xk[-1,0], basin_xk[-1,1], 'xk')

plt.show()



