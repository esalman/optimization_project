import decimal

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