"""
This converter transforms an LP problem from an *.mps file into the matrices for the following form:
min c^T * x
s.t.
Ax <= b
Hx == d

The following assumptions to the problem in the *.mps file are made. If the problem violates one of these assumptions,
the converter should be adjusted to it:
- Minimization problem
- Objective is named "cost"
- Equation names made from pyomo (because of sorting)
- No greater than equation
- Only lower bounds (>= 0) of variables and free bound variables
"""


import numpy as np
from natsort import natsorted
from scipy.sparse import csr_matrix


# return list of all variables and all variables with a lower bound of zero (>= 0)
def getListOfVariablesAndBounds(model_data):
    # area of bounds in mps file
    start_line = model_data.index("BOUNDS\n") + 1
    end_line = model_data.index("ENDATA\n")

    # initialize containers
    all_variables = []
    bounded_variables = []

    for index, line in enumerate(model_data[start_line:end_line]):
        all_variables.append(line.split()[2])

        # only get variables with a lower bound (>= 0)
        sort = line.split()[0]
        if sort == "LO":
            bounded_variables.append(line.split()[2])
        elif sort == "FR":
            pass
        else:
            print(
                "WARNING: Something went wrong with sorting the variables to build variables list and bounding"
                "variables list!"
            )

    # sort the list of variables
    all_variables = natsorted(all_variables)
    bounded_variables = natsorted(bounded_variables)

    # save as dictionary
    all_variables = dict(zip(all_variables, range(0, len(all_variables))))
    bounded_variables = dict(zip(bounded_variables, range(0, len(bounded_variables))))

    return all_variables, bounded_variables


# get data to build matrices c, A and H
def getListsOfcAH(model_data):
    # area of the conditions in mps file
    start_line = model_data.index("COLUMNS\n") + 1
    end_line = model_data.index("RHS\n")

    # initialize containers
    c_cols = []
    c_rows = []
    c_values = []

    A_cols = []
    A_rows = []
    A_values = []

    H_cols = []
    H_rows = []
    H_values = []

    # iterate through every condition
    for index, line in enumerate(model_data[start_line:end_line]):
        # sort the conditions for c (cost-function), A (upper bound) and H (equality)
        sort = line.split()[1][2]
        if sort == "s":
            c_cols.append(line.split()[0])
            c_rows.append(line.split()[1])
            c_values.append(float(line.split()[2]))
        elif sort == "u":
            A_cols.append(line.split()[0])
            A_rows.append(line.split()[1])
            A_values.append(float(line.split()[2]))
        elif sort == "e":
            H_cols.append(line.split()[0])
            H_rows.append(line.split()[1])
            H_values.append(float(line.split()[2]))
        elif sort == "l":
            print("WARNING: Greater than (>=) equation will be ignored!")
        else:
            print("WARNING: Something went wrong with sorting the conditions to build the matrices!")

    # combine containers for each matrix
    c = [c_cols] + [c_rows] + [c_values]
    A = [A_cols] + [A_rows] + [A_values]
    H = [H_cols] + [H_rows] + [H_values]
    return c, A, H


# get data to build vectors b and d
def getListOfbc(model_data):
    # area of RHS (right-hand-side) in mps file
    start_line = model_data.index("RHS\n") + 1
    end_line = model_data.index("BOUNDS\n")

    # initialize containers
    b_rows = []
    b_values = []

    d_rows = []
    d_values = []

    # iterate through RHS definition
    for index, line in enumerate(model_data[start_line:end_line]):
        # sort the condition for b (<= equation) and d (== equation)
        sort = line.split()[1][2]
        if sort == "u":
            b_rows.append(line.split()[1])
            b_values.append(float(line.split()[2]))
        elif sort == "e":
            d_rows.append(line.split()[1])
            d_values.append(float(line.split()[2]))
        elif sort == "l":
            print("WARNING: Greater than (>=) equation will be ignored!")
        else:
            print("WARNING: Something went wrong with sorting the RHS to build the vectors!")

    # combine containers for each matrix
    b = [b_rows] + [b_values]
    d = [d_rows] + [d_values]

    return b, d


# add a constraint to A and b for every variable with a lower bound of zero
def getAbmodified(A_data, b_data, bounded_variables):
    # unpack data
    A_cols = A_data[0]
    A_rows = A_data[1]
    A_values = A_data[2]

    b_rows = b_data[0]
    b_values = b_data[1]

    # iterate through every bounded variable
    for index, bounded_variable in enumerate(bounded_variables):
        # add equation for every bounded variable
        equation_name = "c_u_boundEQ(" + str(index) + ")_"
        A_cols.append(bounded_variable)
        A_rows.append(equation_name)
        A_values.append(float(-1))

        b_rows.append(equation_name)
        b_values.append(float(0))

    # combine containers
    A = [A_cols] + [A_rows] + [A_values]
    b = [b_rows] + [b_values]

    return A, b


# return sparse matrix and description of rows
def getDictAndMatrix(matrix_list, all_variables):
    # unpack data
    cols = matrix_list[0]
    rows = matrix_list[1]
    values = matrix_list[2]

    # get a sorted list of all row names included in the matrix
    # this is later used as a dictionary to assign a unique number to each equation name
    row_names = natsorted(list(set(rows)))
    row_names = dict(zip(row_names, range(0, len(row_names))))

    # convert columns (=> variables) into numbers for further use with csr_matrix
    # all_variables works as a dictionary and converts variable names into a unique number
    cols_matrix = []
    for col in cols:
        cols_matrix.append(all_variables[col])

    # convert rows (=> equations) into numbers for further use with csr_matrix
    # row_names works as a dictionary and converts equation names into a unique number
    rows_matrix = []
    for row in rows:
        rows_matrix.append(row_names[row])

    # build sparse matrix (only filled sections are saved)
    sparse_matrix = csr_matrix((values, (rows_matrix, cols_matrix)), shape=(len(row_names), len(all_variables)))

    return row_names, sparse_matrix


# return vectors as array
def getDictAndVectors(vector_list, row_names):
    # unpack data
    rows = vector_list[0]
    values = vector_list[1]

    # convert rows (=> equation names) into numbers (same as for the associated matrix) and save them in rows_vector
    # row_names works as a dictionary and converts equation names into a unique number
    rows_vector = []
    for row in rows:
        rows_vector.append(row_names[row])

    # sort vector values according to rows_vector (so that it fits to the respective matrix)
    # and return the values in the right order
    values_new = np.array([value for row, value in sorted(zip(rows_vector, values))])

    return values_new


# return all necessary data to fill model
def script(model_data, bounded_variables_as_equations=True):
    # get data out of the mps file
    all_variables, bounded_variables = getListOfVariablesAndBounds(model_data)
    c, A, H = getListsOfcAH(model_data)
    b, d = getListOfbc(model_data)

    if bounded_variables_as_equations:
        # insert new equations for bounded variables into A and b
        A, b = getAbmodified(A, b, bounded_variables)

    # get sparse matrices and row names for each matrix
    c_row_names, c = getDictAndMatrix(c, all_variables)
    A_row_names, A = getDictAndMatrix(A, all_variables)
    H_row_names, H = getDictAndMatrix(H, all_variables)

    # get b and d vector
    # the row names (=> equation names) are defined through the row names of the respective matrix
    b = getDictAndVectors(b, A_row_names)
    d = getDictAndVectors(d, H_row_names)

    return all_variables, bounded_variables, c_row_names, c, A_row_names, A, H_row_names, H, b, d


# should be done in python file where the matrices are needed
def run():
    model_data = open(r"..\source\model.mps", "r").readlines()
    all_variables, bounded_variables, c_row_names, c, A_row_names, A, H_row_names, H, b, d = script(model_data)

    return all_variables, bounded_variables, c_row_names, c, A_row_names, A, H_row_names, H, b, d


# all_variables, bounded_variables, c_row_names, c, A_row_names, A, H_row_names, H, b, d = run()
