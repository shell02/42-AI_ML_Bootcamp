from matrix import Matrix, Vector

print("Let's create some Matrices and Vectors")
m1 = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(m1.shape)

m2 = Matrix((3, 7))
print(m2.shape)

v1 = Vector([[5.0], [3.0], [8.0]])
v2 = Vector([[6.0], [2.0], [7.0]])

#m3 = Matrix('Hello')
#m3 = Matrix([[0.0, 9.0], [5.0, 8.0, 7.0]])
#v2 = Vector([[0.9], [7.0, 8.0], [8.7]])

print("Let's do some operations")
m2 = Matrix([[48.0, 12.0, 3.0], [5.0, 7.0, 16.0]])
m3 = Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

print("Addition:", m1 + m2)
print("Substraction:", m1 - m2)
print("Multiplication of matrices:", m1 * m3)
print("Multiplication of Matrix and Vector:", m1 * v1)
print("Multiplication of Matrix and scalar:", m1 * 3)
print("Division:", m1 / 2)
print("Transpose:", m1.T())
print("Dot product:", v1.dot(v2))
