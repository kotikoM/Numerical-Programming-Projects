import math

def calculate_norm1(v):
    return sum(abs(x) for x in v)

def calculate_norm2(v):
    return math.sqrt(sum(x ** 2 for x in v))

def calculate_norm3(v):
    return max(v)

def calculate_norm4(v):
    return math.pow(sum(math.pow(abs(x), math.pi) for x in v), 1 / math.pi)


vector1 = [100, 2, 3, 4, -20]

norm1 = calculate_norm1(vector1)
print(f"The first norm of the vector is: {norm1}")

norm2 = calculate_norm2(vector1)
print(f"The euclidian norm of the vector is: {norm2}")

norm3 = calculate_norm3(vector1)
print(f"The infinity norm of the vector is: {norm3}")

norm4 = calculate_norm4(vector1)
print(f"The p norm of the vector is: {norm4}")
