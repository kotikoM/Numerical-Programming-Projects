import numpy as np


def input_matrix():
    print("Enter a 3x3 matrix: ")
    values = list(map(float, input().split()))
    if len(values) != 9:
        raise ValueError("You must enter exactly 9 values.")
    matrix = np.array(values).reshape(3, 3)
    print(matrix)
    return matrix


def generate_matrices(n=10):
    matrices = [np.random.uniform(-100, 100, (3, 3)) for _ in range(n)]
    print(matrices)
    return matrices


def calculate_frobenius_norm(matrix):
    sum_of_squares = 0
    for row in matrix:
        for element in row:
            sum_of_squares += element ** 2
    frobenius_norm = sum_of_squares ** 0.5
    return frobenius_norm


def calculate_first_norm(matrix):
    max_sum = 0
    for col in range(matrix.shape[1]):
        col_sum = sum(abs(matrix[row, col]) for row in range(matrix.shape[0]))
        if col_sum > max_sum:
            max_sum = col_sum
    return max_sum

def calculate_infinity_norm(matrix):
    max_sum = 0
    for row in range(matrix.shape[0]):
        row_sum = sum(abs(matrix[row, col]) for col in range(matrix.shape[1]))
        if row_sum > max_sum:
            max_sum = row_sum
    return max_sum

def compare_matrices(matrix1, matrix2, norm_func):
    return norm_func(matrix1 - matrix2)


def main():
    user_matrix = input_matrix()
    random_matrices = generate_matrices()

    min_frobenius_diff = float('inf')
    min_first_norm_diff = float('inf')
    min_infinity_norm_diff = float('inf')

    closest_frobenius_matrix = None
    closest_first_norm_matrix = None
    closest_infinity_norm_matrix = None

    for random_matrix in random_matrices:
        frobenius_diff = abs(compare_matrices(user_matrix, random_matrix, calculate_frobenius_norm))
        first_norm_diff = abs(compare_matrices(user_matrix, random_matrix, calculate_first_norm))
        infinity_norm_diff = abs(compare_matrices(user_matrix, random_matrix, calculate_infinity_norm))

        if frobenius_diff < min_frobenius_diff:
            min_frobenius_diff = frobenius_diff
            closest_frobenius_matrix = random_matrix

        if first_norm_diff < min_first_norm_diff:
            min_first_norm_diff = first_norm_diff
            closest_first_norm_matrix = random_matrix

        if infinity_norm_diff < min_infinity_norm_diff:
            min_infinity_norm_diff = infinity_norm_diff
            closest_infinity_norm_matrix = random_matrix

    print("Matrix with smallest Frobenius norm difference:")
    print(closest_frobenius_matrix)
    print(f"Frobenius norm difference: {min_frobenius_diff}")

    print("\nMatrix with smallest First norm difference:")
    print(closest_first_norm_matrix)
    print(f"First norm difference: {min_first_norm_diff}")

    print("\nMatrix with smallest Infinity norm difference:")
    print(closest_infinity_norm_matrix)
    print(f"Infinity norm difference: {min_infinity_norm_diff}")

if __name__ == '__main__':
    main()


# Enter a 3x3 matrix:
# 1 2 3 4 5 6 7 8 9
# [[1. 2. 3.]
#  [4. 5. 6.]
#  [7. 8. 9.]]
# [array([[ 74.49920684,  -1.97654451, -41.10052781],
#        [-25.45378366, -43.17153827, -16.89486025],
#        [ 54.84956514, -31.30173869, -67.95582382]]),
#  array([[ 48.52655622,  14.67970263,  53.9586121 ],
#        [ 43.33732199, -33.8652095 ,  48.16138944],
#        [-17.72820054,  -6.35536656, -61.86897198]]),
#  array([[-66.00334518, -54.72236481, -97.35964468],
#        [-32.78038025, -44.19887767,  13.82716742],
#        [ 44.15552143,  39.03484654,  97.04675073]]),
#  array([[-54.89671025,  88.00920925, -17.90287856],
#        [ 26.9735072 , -17.67765568,  25.20743733],
#        [ 55.26565598,  17.28963264,  -9.69453911]]),
#  array([[ 44.51205137, -38.13802642,  99.05488101],
#        [-94.20203952, -21.67807647,  -6.9389699 ],
#        [-44.55716105, -63.82138982, -37.98166778]]),
#  array([[-53.82048243, -77.53742435, -82.37572585],
#        [ 23.69826261, -98.306263  ,  -8.50524451],
#        [ 67.68399136, -44.89485694, -24.67291481]]),
#  array([[  7.45459827, -56.14029078,  26.25359512],
#        [-25.88037607,  80.80152843,  41.70222496],
#        [-59.30271202, -50.84153239, -94.46366105]]),
#  array([[-61.67698229, -32.40202824, -17.89220549],
#        [  7.69370444,  86.09185675,  -9.06774274],
#        [ 13.25736337, -36.2981473 ,  18.47070672]]),
#  array([[-90.63124027, -36.55706418,  26.25710187],
#        [ 34.60731697,  90.03377395, -53.50644109],
#        [-59.88019736,  69.9274689 , -43.5767295 ]]),
#  array([[-14.54868438, -49.32088346, -59.41595808],
#        [-69.8409307 ,  69.80179649, -55.8642108 ],
#        [-39.41981311,  25.72726139,  46.01193751]])]
#
# Matrix with smallest Frobenius norm difference:
# [[-61.67698229 -32.40202824 -17.89220549]
#  [  7.69370444  86.09185675  -9.06774274]
#  [ 13.25736337 -36.2981473   18.47070672]]
# Frobenius norm difference: 103.79024776315366
#
# Matrix with smallest First norm difference:
# [[-54.89671025  88.00920925 -17.90287856]
#  [ 26.9735072  -17.67765568  25.20743733]
#  [ 55.26565598  17.28963264  -9.69453911]]
# First norm difference: 119.13587342907101
#
# Matrix with smallest Infinity norm difference:
# [[-61.67698229 -32.40202824 -17.89220549]
#  [  7.69370444  86.09185675  -9.06774274]
#  [ 13.25736337 -36.2981473   18.47070672]]
# Infinity norm difference: 87.97121601426576
#
# As we see in this case the two matrix is closest
# with two of the norms and different one is closer with other norm.
# The difference is calculated by the absolute value of the difference.
# This is good case since two of the three norms are giving same answer.
# Sometimes there may be that all three matrices are different.
# Sometimes all three might be same.
