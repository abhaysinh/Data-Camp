'''

Basics of NetworkX API, using Twitter network

To get you up and running with the NetworkX API, we will run through some basic functions that let you query a Twitter network that has been pre-loaded for you and is available in the IPython Shell as T. The Twitter network comes from KONECT, and shows a snapshot of a subset of Twitter users. It is an anonymized Twitter network with metadata.

You're now going to use the NetworkX API to explore some basic properties of the network, and are encouraged to experiment with the data in the IPython Shell.

Wait for the IPython shell to indicate that the graph that has been preloaded under the variable name T (representing a Twitter network), and then answer the following question:

What is the size of the graph T, the type of T.nodes(), and the data structure of the third element of the last edge listed in T.edges(data=True)? The len() and type() functions will be useful here. To access the last entry of T.edges(data=True), you can use list(T.edges(data=True))[-1].

Instructions
50 XP

Possible Answers

    23369, networkx.classes.reportviews.NodeView, dict.
    32369, tuple, datetime.
    23369, networkx.classes.reportviews.NodeView, datetime.
    22339, dict, dict.

Answer :  23369, networkx.classes.reportviews.NodeView, dict.
'''