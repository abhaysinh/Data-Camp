'''
String Methods

Strings come with a bunch of methods. Follow the instructions closely to discover some of them.
If you want to discover them in more detail, you can always type help(str) in the IPython Shell.

A string place has already been created for you to experiment with.
Instructions
100 XP

    1   Use the upper() method on place and store the result in place_up. Use the syntax for calling methods
    that you learned in the previous video.

    2   Print out place and place_up. Did both change?

    3   Print out the number of o's on the variable place by calling count() on place and passing the letter 'o'
    as an input to the method. We're talking about the variable place, not the word "place"!

'''

# string to experiment with: place
place = "poolhouse"

# Use upper() on place: place_up
place_up = place.upper()

# Print out place and place_up
print(place)
print(place_up)

# Print out the number of o's in place
print(place.count('o'))