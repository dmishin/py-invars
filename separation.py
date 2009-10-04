import numpy
from matplotlib import pyplot
import random
import math

x=numpy.linspace(0,1,100)
y=numpy.sin(x)


textdata = """
xxx  
  x x xx x             yyy  y 
   xx x x x             yyyy
    xxx  x x x x        yy 
       x x xx   x x      yyyyyyyyy
           xx xxx    x x     
            xx x yyyy  xxxx
            xxx  yyyy xxx 
              xxx     xxx
                xx
                 x
                 xxx
                 x x
                 x
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
    numpy.vectorize(lambda x:math.atan(x))
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


def test():
    random.seed(1001)
    data = convert_text_data(textdata)
    normalize_points(*(data.values()))
    plot_points(*(data.values()))
    pyplot.axis([-1.1,1.1,-1.1,1.1])
    pyplot.show()


test()
