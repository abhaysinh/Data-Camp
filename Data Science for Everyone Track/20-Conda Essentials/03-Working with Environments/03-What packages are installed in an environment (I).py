'''
What packages are installed in an environment? (I)

The command conda list seen previously displays all packages installed in the current environment. You can reduce this list by appending the particular package you want as an option. The package can be specified either as a simple name, or as a regular expression pattern. This still displays the version (and channel) associated with the installed package(s). For example:

(test-env) $ conda list 'numpy|pandas'
# packages in environment at /home/repl/miniconda/envs/test-env:
#
# Name                    Version                   Build  Channel
numpy                     1.11.3                   py35_0
pandas                    0.18.1              np111py35_0

Without specifying 'numpy|pandas', these same two lines would be printed, simply interspersed with many others for the various other installed packages. Notice that the output displays the filepath associated with the current environment first: in this case, /home/repl/miniconda/envs/test-env as test-env is the active environment (as also shown at the prompt).

Following this example, what versions of numpy and pandas are installed in the current (base/root) environment?

Instructions
50 XP

Possible Answers

    numpy=1.11.3; pandas=0.18.1
    numpy=1.16.0; pandas=0.22.0
    numpy=2.0.1; pandas=0.22.2
    numpy=1.10.4; pandas=0.17.1
    numpy=1.13.1; pandas=0.21.0
    numpy=1.15.0; pandas 0.23.0

Answer :     numpy=1.16.0; pandas=0.22.0
'''