'''
Which environment am I using?

When using conda, you are always in some environment, but it may be the default (called the base or root environment). Your current environment has a name and contains a collection of packages currently associated with that environment. There are a few ways to determine the current environment.

Most obviously, at a terminal prompt, the name of the current environment is usually prepended to the rest of your prompt in parentheses. Alternatively, the subcommand conda env list displays a list of all environments on your current system; the currently activated one is marked with an asterisk in the middle column. The subcommands of conda env (sometimes with suitable switches) encompass most of your needs for working with environments.

The output of conda env list shows that each environment is associated with a particular directory. This is not the same as your current working directory for a given project; being "in" an environment is completely independent of the directory you are working in. Indeed, you often wish to preserve a certain Conda environment and edit resources across multiple project directories (all of which rely on the same environment). The environment directory displayed by conda env list is simply the top-level file path in which all resources associated with that environment are stored; you need never manipulate those environment directories directly (other than via the conda command); indeed, it is much safer to leave those directories alone!

For example, here is output you might see in a particular terminal:

(test-project) $ conda env list
# conda environments:
#
base                     /home/repl/miniconda
py1.0                    /home/repl/miniconda/envs/py1.0
stats-research           /home/repl/miniconda/envs/stats-research
test-project          *  /home/repl/miniconda/envs/test-project

Following the example above, what is the name of the environment you are using in the current session? Even if you determine the answer without running a command, run conda env list to get a feel of using that subcommand.

Instructions
50 XP

Possible Answers

    base
    test-project
    root
    course-project
    stats-research
    py1.0

Answer :  course-project
'''