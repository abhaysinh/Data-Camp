'''
Compute degree distribution

The number of neighbors that a node has is called its "degree", and it's possible to compute the degree distribution across the entire graph. In this exercise, your job is to compute the degree distribution across T.

Instructions
100 XP

    Use a list comprehension along with the .neighbors(n) method to get the degree of every node. The result should be a list of integers.
        Use n as your iterator variable.
        The output expression of your list comprehension should be the number of neighbors that node n has - that is, its degree. Use the len() and list() functions together with the .neighbors() method to compute this.
        The iterable in your list comprehension is all the nodes in T, accessed using the .nodes() method.
    Print the degrees.

'''


# Compute the degree of every node: degrees
degrees = [len(list(T.neighbors(n))) for n in T.nodes()]

# Print the degrees
print(degrees)