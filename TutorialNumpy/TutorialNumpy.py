import numpy as np

a = np.array([1,2,3])
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])
a[0] = 5
print(a)

b = np.array([[1,2,3],[4,5,6]])
print(b.shape)
print(b[0,0], b[0,1], b[1,0])

a = np.zeros((2,2))
print(a)

b=np.ones((1,2))
print(b)

c = np.full((2,2),7)
print(c)

d = np.eye(2)
print(d)

e = np.random.random((2,2))
print(e)

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

b = a[:2, 1:3]

print(a[0,1])
b[0,0] = 77
print(a[0,1])

print(r'np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])')
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print("rowR1 = a[1,:] \n rowR2 = a[1:2, :]")
rowR1 = a[1,:]
rowR2 = a[1:2, :]
print(rowR1, rowR1.shape)
print(rowR2, rowR2.shape)


print("colR1 = a[:,1] \n colR2 = a[:,1:2]")
colR1 = a[:,1]
colR2 = a[:,1:2]

print(colR1, colR1.shape)
print(colR2, colR2.shape)

#Integer array indexing: When you index into numpy arrays using slicing, the resulting array view will 
#always be a subarray of the original array. In contrast, integer array indexing allows you to 
#construct arbitrary arrays using the data from another array. Here is an example:

a = np.array([[1,2], [3, 4], [5, 6]])

print("a = ", a)
# An example of integer array indexing.
# The returned array will have shape (3,) and
print(r'a[[0,1,2],[0,1,0]] :')
print(a[[0,1,2],[0,1,0]]) # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0,0],[1,1]]) #Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array(a[0,1], a[0,1])) #Prints "[2 2]"

#######
#One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)

#Create an array of indices
b = np.array([0,2,0,1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b]) # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10
print(a)
# prints "array([[11,  2,  3],
#                [ 4,  5, 16],
#                [17,  8,  9],
#                [10, 21, 12]])


#Boolean array indexing: Boolean array indexing lets you pick out arbitrary elements of an array.
# Frequently this type of indexing is used to select the elements of an array that satisfy some condition.
#  Here is an example:

a = np.array([[1,2], [3, 4], [5, 6]])
boolIdx = (a>2)

# Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.
print(boolIdx)
# Prints "[[False False]
#          [ True  True]
#          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[boolIdx]) #Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a>2]) #Prints "[3 4 5 6]"

#DATATYPES
x = np.array([1,2])
print(x.dtype)

x=np.array([1.0,2.0])
print(x.dtype)

x=np.array([1,2], dtype=np.float64)
print(x.dtype)


#ARRAY MATH

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]

print(x + y)
print(np.add(x,y))

# Elementwise difference; both produce the array
print(x + y)
print(np.substract(x,y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x*y)
print(np.multiply(x,y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x/y)
print(np.divide(x,y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))


#Note that unlike MATLAB, * is elementwise multiplication, not matrix multiplication. 
#We instead use the dot function to compute inner products of vectors, to multiply a vector by a matrix, 
#and to multiply matrices. dot is available both as a function in the numpy module and as an instance method of array objects:

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print("# Inner product of vectors; both produce 219")
print(v.dot(w))
print(np.dot(v,w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print("# Matrix / vector product; both produce the rank 1 array [29 67]")
print(x.dot(v))
print(np.dot(x,v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print("# Matrix / matrix product; both produce the rank 2 array  [[19 22] [43 50]]")
print(x.dot(y))
print(np.dot(x,y))

#Numpy provides many useful functions for performing computations on arrays; one of the most useful is sum:

x = np.array([[1,2],[3,4]])

print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sup(x, axis=1))

