# 1. Simple Addition of Small and Large Numbers
large_number = 1.0
small_number = 1e-16
result_1 = large_number + small_number
print(f"Example 1: {large_number} + {small_number} = {result_1}")
# Example 1: 1.0 + 1e-16 = 1.0

# 2. Repeating Decimals
repeating_decimal = 1 / 3
print(f"Example 2: 1/3 = {repeating_decimal:.30f}")  # Show 10 decimal places
# Example 2: 1/3 = 0.333333333333333314829616256247

# 3. Addition/Subtraction Precision Loss
result_3 = 1.0000001 - 1.0000000
print(f"Example 3: 1.0000001 - 1.0000000 = {result_3}")
# Example 3: 1.0000001 - 1.0000000 = 1.0000000005838672e-07

# 4. Non-Associative Arithmetic with Smaller Values
a = 1e16
b = -1e16
c = 1.0
result1 = (a + b) + c
result2 = a + (b + c)
print(f"Example 4: ((a + b) + c = {result1})")
print(f"Example 4: (a + (b + c) = {result2})")
# Example 4: ((a + b) + c = 1.0)
# Example 4: (a + (b + c) = 0.0)

