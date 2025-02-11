def f(x, y):
    return 3 * (x ** 2) + 4 * (y ** 2) - (x * y) + (5 * x) - (3 * y) + 7


def gradient_x(x, y):
    return (6 * x) - y + 5


def gradient_y(x, y):
    return (8 * y) - x - 3


x, y = 1, 1
eta = 0.05

for i in range(3):
    grad_x = gradient_x(x, y)
    grad_y = gradient_y(x, y)

    x = x - eta * grad_x
    y = y - eta * grad_y
    print(f"After updating x and y: Iteration {i + 1}")
    print(f"x: {x}, y: {y}, f(x, y): {f(x, y)}")
    print("------------------------------")


# After updating x and y: Iteration 1
# x: 0.5, y: 0.8, f(x, y): 10.01
# ------------------------------
# After updating x and y: Iteration 2
# x: 0.13999999999999996, y: 0.655, f(x, y): 7.4182
# ------------------------------
# After updating x and y: Iteration 3
# x: -0.11925000000000002, y: 0.55, f(x, y): 6.0719991874999995
# ------------------------------
