import numpy
from matplotlib import pyplot
import random
import math
import scipy.linalg

x=numpy.linspace(0,1,100)
y=numpy.sin(x)


textdata1 = """
xxx  
  x x xx x              yy  y 
   xx x x x             yyyy
    xxx  x x x          yy 
       x x xx            yyyyyyyyy
                             
"""
textdata = """
                        yyyy
                         yy
                       yyyyy

xxx  xxxxxxxxx xxxxxxxxxxxxxxxxxx  xxxxxxxxxxxxxx
                              
                               x             xxx
                        yy 
                         yyyyyyyyy       xx 
                         
          xxxx                            xx
          xx  xx                         x x
           xxxxxx xxxxxxxxxxxxxxxx x x x
                                                       yyyyyyyyyyyyyyyy
"""

def convert_text_data(text, x0=0, dx=0.1, y0=0, dy=-0.1, randomize = True):
    """Read the ascii-art text string and produce the coordinate vectors for each non-space character
    """
    vectors = dict()

    for Y,line in enumerate(text.split("\n")):
        for X,char in enumerate(line):
            if char !=' ':
                x = X*dx+x0
                y = Y*dy+y0
                if randomize:
                    x += random.random()*dx
                    y += random.random()*dy
                    
                try:
                    vec = vectors[char]
                except KeyError:
                    vec = [list(), list()]
                    vectors[char] = vec
                vec[0].append(x)
                vec[1].append(y)
    for char, vec in vectors.items():
        vectors[char] = numpy.array(vec)

    return vectors

def plot_points(*point_arrays):
    """Plot the arrays of points, each point is a column in the array. """
    plot_args=[]
    for points in point_arrays:
        plot_args.append(points[0])
        plot_args.append(points[1])
        plot_args.append("o")
    return pyplot.plot(*plot_args)
    


expansion_functions = [
    numpy.vectorize(lambda x: x*x),
    #numpy.vectorize(lambda x:math.atan(x)),
    #numpy.cos,
    #numpy.abs,
    ]

def nonlinear_expand(vectors):
    """Expand dimension of the vectors, applying some nonlinear functions to its coordinates"""
    st=[vectors]
    for func in expansion_functions:
        st.append(func(vectors))
    return numpy.vstack(st)


def scale_vectors(vectors, scale):
    """Same as vectors := diag(scale)*vectors"""
    assert(vectors.shape[0]==len(scale))
    for i in xrange(vectors.shape[0]):
        vectors[i,:] *= scale[i]
        
def offset_vectors(vectors, offset):
    assert(vectors.shape[0]==len(offset))
    for i in xrange(vectors.shape[0]):
        vectors[i,:] += offset[i]
    

#print nonlinear_expand(data['x'])

def normalize_points(*point_arrays):
    """Given several array of points, normalize them so that all coordinates was in the [-1..+1] diapasone
    Normalization is done separately by every coordinate.
    """
    #maximums and minimums by each coordinate
    maximums = numpy.array([points.max(1) for points in point_arrays]).max(0)
    minimums = numpy.array([points.min(1) for points in point_arrays]).min(0)
    
    means  = (maximums+minimums)*0.5
    ranges = (maximums-minimums)*0.5
    
    for points in point_arrays:
        offset_vectors(points, -means)
        scale_vectors(points, 1.0/ranges)
    return means, ranges




def mean_a_at (V):
    """ 1/n * sum( A_i*A_I^T )
where A_i is i'th column vector.
Result is square symmetric matrix of the same size as the vector dimensoin
V must be an numpy array
"""
    assert(isinstance(V, numpy.array))
    n,N = V.shape #n - size of the vector, N - number of vectors
    M = numpy.dot(V*V.T)*(1.0/N)
    return M

def column(vec):
    """Convert 1-d vector to the column vector"""
    shp = vec.shape
    if len(shp) != 1: raise ValueError, "matrix must b 1D numpy array"
    rval = numpy.array(vec, copy = False)
    rval.shape = shp+(1,)
    return rval
    
def calculate_d_matrix(A,B):
    """D = A*A' + B*B' - sum(Ai)*sum(Bi)' - sum(Bi)*sum(Ai)'
Main eigenvector of this matrix is the directin, in which vector clouds A and B are most distinguishable"""
    mA = column(numpy.mean(A, 1)) #sum of all vectors A
    mB = column(numpy.mean(B, 1))
    
    D =   numpy.dot(A,A.T)/A.shape[1] \
        + numpy.dot(B,B.T)/B.shape[1] \
        - numpy.dot(mA,mB.T)          \
        - numpy.dot(mB,mA.T)
    return D

def test():
    M = 2 #size of the reduced vector
    
    #prepare data
    random.seed(1001)
    data = convert_text_data(textdata)
    X = data['x']
    Y = data['y']
    
    #normalize data
    normalize_points(X,Y)
    
    for iter in range(30):
        print "========================================================"
        print "Iteration", iter
        #expansion
        Xe = nonlinear_expand(X)
        Ye = nonlinear_expand(Y)
        
        #calculate the best separation for the expanded points:
        lam_e, h_e = numpy.linalg.eigh( calculate_d_matrix(Xe,Ye) )
        #columns of H are eigenvectors
        print "Lambda for the expanded points:", lam_e
        #get the M highest eigenvectors and their eigenvalues
        lam_h = [(lam_e[i], h_e[:,i:(i+1)]) for i in xrange(len(lam_e))]
        lam_h.sort(key=lambda x:x[0])
        lam_h = lam_h[-M:]
        print "Principial lambda:", [l for l,h in lam_h]
        h_red = numpy.hstack([h for l,h in lam_h])
        print "Principial eigenvectors:\n", h_red
        
        #reducing the dimension
        Xe_red = numpy.dot(h_red.T, Xe)
        Ye_red = numpy.dot(h_red.T, Ye)
        
        #normalize points
        normalize_points(Xe_red,Ye_red)
        
            
        plot_points(Xe_red, Ye_red)
        pyplot.axis([-2,2,-2,2])
        pyplot.show()
        
        raw_input("Press enter")
        pyplot.close()
        
        #and repeat the iteration
        X = Xe_red
        Y = Ye_red
        
def test1():
    "Testing with modified algorithm"
    M = 3 #size of the reduced vector
    
    #prepare data
    random.seed(1001)
    data = convert_text_data(textdata)
    X = data['x']
    Y = data['y']
    pyplot.figure(1)
    
    #normalize data
    normalize_points(X,Y)
    
    for iter in range(30):
        print "========================================================"
        print "Iteration", iter
        #expansion
        A = nonlinear_expand(X)
        B = nonlinear_expand(Y)
        na = A.shape[1]
        nb = B.shape[1]
        
        #calculate the best separation for the expanded points:
        R = numpy.dot(A,A.T)*(1.0/na) + numpy.dot(B,B.T)*(1.0/nb)
        A0 = column(A.mean(1))
        B0 = column(B.mean(1))
        P = R - numpy.dot(A0,B0.T) - numpy.dot(B0, A0.T)
        Q = R - numpy.dot(A0,A0.T) - numpy.dot(B0, B0.T)
        
        #print "lam P", numpy.linalg.eigh(P)[0]
        #print "lam Q", numpy.linalg.eigh(Q)[0]
        
        # Px = lambda*Qx
        lam, H = scipy.linalg.eigh(P, Q)
        
        #test the criteria
        for col in xrange(len(H)):
            hi = H[:,col:(col+1)]
            k = numpy.dot(numpy.dot(hi.T,P),hi)[0,0] / numpy.dot(numpy.dot(hi.T,Q),hi)[0,0]
            print "lambda=",lam[col], "K(hi)=",k
        
        #columns of H are eigenvectors
        print "Lambda for the expanded points:", lam
        #get the M highest eigenvectors and their eigenvalues
        H = H[:, (H.shape[1]-1):(H.shape[1]-1-M):(-1)]
        print "Principial eigenvectors:\n", H
        
        #reducing the dimension
        X = numpy.dot(H.T, A)
        Y = numpy.dot(H.T, B)
        
        #normalize points
        normalize_points(X, Y)
        
            
        plot_points(X, Y)
        pyplot.axis([-2,2,-2,2])
        pyplot.show()
        
        raw_input("Press enter")
        pyplot.close()
        
        #and repeat the iteration

test1()


