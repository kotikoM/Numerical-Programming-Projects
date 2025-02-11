# 1. Subtracting Very Small Numbers
small_a = 1e-10
small_b = 1e-20
result_1 = small_a - small_b
print(f"Example 1: {small_a} - {small_b} = {result_1:.20f}\n")
# Example 1: 1e-10 - 1e-20 = 0.00000000009999999999

# 2. Multiplying Small Numbers
small_a = 1e-200
small_b = 1e-200
result_2 = small_a * small_b
print(f"Example 2: {small_a} * {small_b} = {result_2:.2000f}\n")
# Example 2: 1e-200 * 1e-200 = 0.00000000000000000000

# 3. Adding Large and Small Numbers
large_number = 1e10
small_number = 1e-10
result_4 = large_number + small_number
print(f"Example 3: {large_number} + {small_number} = {result_4:.20f}\n")
# Example 3: 10000000000.0 + 1e-10 = 10000000000.00000000000000000000

# 4. Exponential Function with Large Negative Input
import math

small_value = -1000  # A very small value for exp
result_5 = math.exp(small_value)
print(f"Example 4: exp({small_value}) = {result_5:.20f}")  # This will be close to 0
# Example 4: exp(-1000) = 0.00000000000000000000
