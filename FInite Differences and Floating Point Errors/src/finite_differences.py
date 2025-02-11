import math

def f(x):
    return math.sin(x)

def df(x):
    return math.cos(x)



def forward_difference(x, h):
    return (f(x + h) - f(x)) / h


def backward_difference(x, h):
    return (f(x) - f(x - h)) / h


def central_difference(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def richardsons_extrapolation(x, h):
    return (4 * central_difference(x, h / 2) - central_difference(x, h)) / 3
