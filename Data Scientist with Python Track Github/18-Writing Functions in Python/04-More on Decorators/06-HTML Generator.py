'''

HTML Generator

You are writing a script that generates HTML for a webpage on the fly. So far, you have written two
decorators that will add bold or italics tags to any function that returns a string.
You notice, however, that these two decorators look very similar. Instead of writing a bunch of other
similar looking decorators, you want to create one decorator, html(), that can take any pair of opening and closing tags.

def bold(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    msg = func(*args, **kwargs)
    return '<b>{}</b>'.format(msg)
  return wrapper

def italics(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    msg = func(*args, **kwargs)
    return '<i>{}</i>'.format(msg)
  return wrapper

Instructions

    1   Return the decorator and the decorated function from the correct places in the new html() decorator.  - 25 XP

    2   Use the html() decorator to wrap the return value of hello() in <b> and </b> (the HTML tags that mean "bold").  - 25 XP

    3   Use html() to wrap the return value of goodbye() in <i> and </i> (the HTML tags that mean "italics").   - 25 XP

    4   Use html() to wrap hello_goodbye() in a DIV, which is done by adding <div> and </div> tags around a string.   - 25 XP

'''


# No 1
def html(open_tag, close_tag):
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      msg = func(*args, **kwargs)
      return '{}{}{}'.format(open_tag, msg, close_tag)
    # Return the decorated function
    return wrapper
  # Return the decorator
  return decorator


# No 2
# Make hello() return bolded text
@html('<b>', '</b>')
def hello(name):
  return 'Hello {}!'.format(name)

print(hello('Alice'))


# No 3
# Make goodbye() return italicized text
@html('<i>', '</i>')
def goodbye(name):
    return 'Goodbye {}.'.format(name)


print(goodbye('Alice'))

# No 4
# Wrap the result of hello_goodbye() in <div> and </div>
@html('<div>', '</div>')
def hello_goodbye(name):
    return '\n{}\n{}\n'.format(hello(name), goodbye(name))


print(hello_goodbye('Alice'))