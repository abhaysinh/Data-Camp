'''
Remove an environment

From time to time, it is worth cleaning up the environments you have accumulated just to make management easier. Doing so is not pressing; as they use little space or resources. But it's definitely useful to be able to see a list of only as many environments as are actually useful for you.

The command to remove an environment is:

conda env remove --name ENVNAME

You may also use the shorter -n switch instead.

Instructions
100 XP

    The current session has an environment named deprecated. Remove it from the session.

Solution : 

conda env remove --name deprecated
'''