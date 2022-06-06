from copy import deepcopy

class Matrix(object):
    """
    Creates Matrix and allows both matrix-matrix and matrix-vector
    operations.
    Initialize with a list of lists or a shape.
    """
    def __init__(self, input):
        if type(input) == tuple and len(input) == 2:
            if input[0] > 0 or input[1] > 0:
                self.shape = input
                self.data = [[0 for i in range(input[1])] for j in range (input[0])]
            else:
                raise Exception("Initialization requires a shape of positive numbers")
        elif type(input) == list and len(input) > 0 and type(input[0]) == list:
            base_len = len(input[0])
            for row in input:
                if len(row) != base_len:
                    raise Exception("Initialization requires lists(row) of same lenght")
                self.shape = (len(input), base_len)
                self.data = deepcopy(input)
        else:
            raise Exception("Initialization requires a shape of 2 numbers or a list of lists")

    def __add__(self, matrix):
        """
        Addition of two Matrices.
        """
        if type(matrix) != Matrix or matrix.shape != self.shape:
            raise Exception("Addition requires two matrices of same shape.")
        result = []
        for row in range(self.shape[0]):
            n_row = []
            for col in range(self.shape[1]):
                n_row.append(self.data[row][col] + matrix.data[row][col])
            result.append(n_row)
        return Matrix(result)

    def __radd__(self, matrix):
        """
        Addition of two matrices.
        """
        return self + matrix

    def __sub__(self, matrix):
        """
        Substraction of two Matrices.
        """
        if type(matrix) != Matrix or matrix.shape != self.shape:
            raise Exception("Substraction requires two matrices of same shape.")
        result = []
        for row in range(self.shape[0]):
            n_row = []
            for col in range(self.shape[1]):
                n_row.append(self.data[row][col] - matrix.data[row][col])
            result.append(n_row)
        return Matrix(result)

    def __rsub__(self, matrix):
        """
        Substraction of two Matrices.
        """
        if type(matrix) != Matrix or matrix.shape != self.shape:
            raise Exception("Substraction requires two matrices of same shape.")
        result = []
        for row in range(self.shape[0]):
            n_row = []
            for col in range(self.shape[1]):
                n_row.append(matrix.data[row][col] - self.data[row][col])
            result.append(n_row)
        return Matrix(result)

    def __truediv__(self, scalar):
        """
        Division of Matrix by scalar.
        """
        if type(scalar) != int and type(scalar) != float:
            raise Exception("Division of matrix requires a scalar (int or float).")
        result = []
        for row in range(self.shape[0]):
            n_row = []
            for col in range(self.shape[1]):
                n_row.append(self.data[row][col] / scalar)
            result.append(n_row)
        return Matrix(result)

    def __rtruediv__(self, scalar):
        """
        Division of scalar by Matrix.
        """
        if type(scalar) != int and type(scalar) != float:
            raise Exception("Division of matrix requires a scalar (int or float).")
        result = []
        for row in range(self.shape[0]):
            n_row = []
            for col in range(self.shape[1]):
                n_row.append(scalar / self.data[row][col])
            result.append(n_row)
        return Matrix(result)

    def __mul__(self, term):
        """
        Multiplication of Matrix by Matrix, Vector or scalar.
        """
        result = []
        if type(term) == int or type(term) == float:
            for row in range(self.shape[0]):
                n_row = []
                for col in range(self.shape[1]):
                    n_row.append(self.data[row][col] * term)
                result.append(n_row)
        elif type(term) == Matrix or type(term) == Vector:
            if self.shape[1] != term.shape[0]:
                raise Exception("Multiplication of Matrices requires the same number of rows for one as columns for the other.")
            for row in range(self.shape[0]):
                n_row = []
                for t_col in range(term.shape[1]):
                    value = 0
                    for col in range(self.shape[1]):
                        value += self.data[row][col] * term.data[col][t_col]
                    n_row.append(value)
                result.append(n_row)
        else:
            raise Exception("Multiplication of Matrix requires a Vector, Matrix or scalar.")
        if type(term) == Vector:
            return Vector(result)
        else:
            return Matrix(result)

    def __rmul__(self, term):
        """
        Multiplication of Matrix by Matrix, Vector or scalar.
        """
        return term * self

    def __str__(self):
        """
        Print Matrix data.
        """
        txt = "Matrix(" + str(self.data) + ")"
        return txt

    def __repr__(self):
        """
        Represent Matrix object.
        """
        txt = "Matrix(" + str(self.data) + ")"
        return txt
    
    def T(self):
        """
        Transpose Matrix object.
        """
        result = []
        for col in range(self.shape[1]):
            n_row = []
            for row in range(self.shape[0]):
                n_row.append(self.data[row][col])
            result.append(n_row)
        return Matrix(result)

class Vector(Matrix):
    """
    Creates Vector that inherits from Matrix class.
    A Vector has a dimension of (1, n) or (n, 1).
    """
    def __init__(self, input):
        if type(input) == list and \
            (len(input) == 0 or (len(input) != 1 and len(input[0]) != 1)) :
            raise Exception("A Vector has a dimension of (1, n) or (n, 1).")
        if type(input) == tuple and \
            (len(input) != 2 or (input[0] != 1 and input[1] != 1)):
            raise Exception("A Vector has a dimension of (1, n) or (n, 1).")
        super().__init__(input)
    

    def __add__(self, vector):
        """
        Addition of two Vectors.
        """
        return Vector((Matrix(self.data) + Matrix(vector.data)).data)

    def __radd__(self, vector):
        """
        Addition of two Vectors.
        """
        return self + vector

    def __sub__(self, vector):
        """
        Subtraction of two Vectors.
        """
        return Vector((Matrix(self.data) - Matrix(vector.data)).data)

    def __rsub__(self, vector):
        """
        Subtraction of two Vectors.
        """
        return Vector((Matrix(vector.data) - Matrix(self.data)).data)

    def __truediv__(self, scalar):
        """
        Division of Vector by scalar.
        """
        return Vector((Matrix(self.data) / scalar).data)

    def __rtruediv__(self, scalar):
        """
        Division of scalar by Vector.
        """
        return Vector((scalar / Matrix(self.data)).data)

    def __mul__(self, value):
        """
        Multiplication of Vector by Vector, Matrix or scalar.
        """
        return Vector((Matrix(self.data) * value).data) 

    def __rmul__(self, value):
        """
        Multiplication of Vector by Vector, Matrix or scalar.
        """
        return  Vector((value * Matrix(self.data)).data) 

    def __str__(self):
        """
        Prints Vector data.
        """
        txt = "Vector(" + str(self.data) + ")"
        return txt
    
    def dot(self, vector):
        """
        Dot product of two Vectors
        """
        if type(vector) != Vector:
            raise Exception("Vector dot product requires two vectors.")
        if self.shape != vector.shape:
            raise Exception("Vector dot product requires two vectors of same shape.")
        result = []
        for row in range(self.shape[0]):
            n_row = []
            for col in range(self.shape[1]):
                n_row.append(self.data[row][col] * vector.data[row][col])
            result.append(n_row)
        return result
