'''

Preserving docstrings when decorating functions

Your friend has come to you with a problem. They've written some nifty decorators and added them to the
functions in the open-source library they've been working on. However, they were running some tests and
discovered that all of the docstrings have mysteriously disappeared from their decorated functions.
Show your friend how to preserve docstrings and other metadata when writing decorators.

Instructions

    1   Decorate print_sum() with the add_hello() decorator to replicate the issue that your friend saw -
        that the docstring disappears.               - 25 XP

    2   To show your friend that they are printing the wrapper() function's docstring,
        not the print_sum() docstring, add the following docstring to wrapper():
        """Print 'hello' and then call the decorated function."""   - 25 XP

    3   Import a function that will allow you to add the metadata from print_sum() to the decorated version of print_sum(). - 25 XP

    4   Finally, decorate wrapper() so that the metadata from func() is preserved in the new decorated function.   - 25 XP

'''


# No 1
def add_hello(func):
    def wrapper(*args, **kwargs):
        print('Hello')
        return func(*args, **kwargs)

    return wrapper


# Decorate print_sum() with the add_hello() decorator
@add_hello
def print_sum(a, b):
    """Adds two numbers and prints the sum"""
    print(a + b)


print_sum(10, 20)
print(print_sum.__doc__)


# No 2
def add_hello(func):
    # Add a docstring to wrapper
    def wrapper(*args, **kwargs):
        """Print 'hello' and then call the decorated function."""
        print('Hello')
        return func(*args, **kwargs)

    return wrapper


@add_hello
def print_sum(a, b):
    """Adds two numbers and prints the sum"""
    print(a + b)


print_sum(10, 20)
print(print_sum.__doc__)

# No 3
# Import the function you need to fix the problem
from functools import wraps


def add_hello(func):
    def wrapper(*args, **kwargs):
        """Print 'hello' and then call the decorated function."""
        print('Hello')
        return func(*args, **kwargs)

    return wrapper


@add_hello
def print_sum(a, b):
    """Adds two numbers and prints the sum"""
    print(a + b)


print_sum(10, 20)
print(print_sum.__doc__)


# No 4
from functools import wraps


def add_hello(func):
    # Decorate wrapper() so that it keeps func()'s metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Print 'hello' and then call the decorated function."""
        print('Hello')
        return func(*args, **kwargs)

    return wrapper


@add_hello
def print_sum(a, b):
    """Adds two numbers and prints the sum"""
    print(a + b)


print_sum(10, 20)
print(print_sum.__doc__)