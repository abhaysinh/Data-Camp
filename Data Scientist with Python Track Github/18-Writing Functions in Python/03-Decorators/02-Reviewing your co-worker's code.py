'''

Reviewing your co-worker's code

Your co-worker is asking you to review some code that they've written and give them some
tips on how to get it ready for production. You know that having a docstring is considered best practice for
maintainable, reusable functions, so as a sanity check you decide to run this has_docstring() function on all of their functions.

def has_docstring(func):
  """Check to see if the function
  `func` has a docstring.

  Args:
    func (callable): A function.

  Returns:
    bool
  """
  return func.__doc__ is not None

Instructions


    1   Call has_docstring() on your co-worker's load_and_plot_data() function.    - 35 XP

    2   Check if the function as_2D() has a docstring.      - 35 XP

    3   Check if the function log_product() has a docstring.        - 30 XP

'''

# No 1
# Call has_docstring() on the load_and_plot_data() function
ok = has_docstring(load_and_plot_data)

if not ok:
  print("load_and_plot_data() doesn't have a docstring!")
else:
  print("load_and_plot_data() looks ok")


# No 2
# Call has_docstring() on the as_2D() function
ok = has_docstring(as_2D)

if not ok:
  print("as_2D() doesn't have a docstring!")
else:
  print("as_2D() looks ok")


# No 3
# Call has_docstring() on the log_product() function
ok = has_docstring(log_product)

if not ok:
  print("log_product() doesn't have a docstring!")
else:
  print("log_product() looks ok")