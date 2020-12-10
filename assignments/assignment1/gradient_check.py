import numpy as np

def _check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    #import ipdb; ipdb.set_trace()
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()
    
    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = 0

        # TODO compute value of numeric gradient of f to idx
        
        #x = orig_x.copy()
        #x[ix] = x[ix] + delta
        #numeric_grad_at_ix = (f(x)[0] - f(orig_x)[0])/ delta
        zeros = np.zeros(x.shape)
        zeros[ix] = delta
        numeric_grad_at_ix = (f(np.add(x, zeros))[0] - f(np.subtract(x, zeros))[0])/ (2 * delta)
        print('num', numeric_grad_at_ix)
        
        
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True

def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    #import ipdb; ipdb.set_trace()
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    print(f'Analytic grad:\n{analytic_grad}')
    print(f'Numerical grad:')
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        
        zeros = np.zeros(x.shape)
        zeros[ix] = delta
        
        x_add = np.add(x, zeros)

        numeric_grad_at_ix = (f(x_add)[0] - f(orig_x)[0])/ delta
        
        print(f'{ix}: {numeric_grad_at_ix:.8f}')
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()
    print("Gradient check passed!")
    return True

        

        
