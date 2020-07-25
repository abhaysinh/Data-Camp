'''
List Methods (2)

Most list methods will change the list they're called on. Examples are:

    append(), that adds an element to the list it is called on,
    remove(), that removes the first element of a list that matches the input, and
    reverse(), that reverses the order of the elements in the list it is called on.

You'll be working on the list with the area of different parts of the house: areas.

Instructions
100 XP

    1   Use append() twice to add the size of the poolhouse and the garage again: 24.5 and 15.45, respectively.
        Make sure to add them in this order.

    2   Print out areas

    3   Use the reverse() method to reverse the order of the elements in areas.

    4   Print out areas once more.

'''


# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Use append twice to add poolhouse and garage size
areas.append(24.5)
areas.append(15.45)


# Print out areas
print(areas)

# Reverse the orders of the elements in areas
areas.reverse()

# Print out areas
print(areas)
