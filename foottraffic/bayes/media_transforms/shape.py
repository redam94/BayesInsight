

def hill_function(x, n, k):
    """Hill function S Curve function. If n=1 it is a concave function, 
    if n>1 it is a s curve function.
    """
    return x ** n / (x ** n + k ** n)
  
