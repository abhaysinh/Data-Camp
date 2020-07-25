'''
What version of conda do I have?

The tool conda takes a variety of commands and arguments. Most of the time, you will use conda COMMAND OPTIONS --SWITCH. You will learn the collection of COMMANDs available in the next lessons. A summary is available on the help screen:

$ conda --help
usage: conda [-h] [-V] command ...

conda is a tool for managing and deploying applications, environments and packages.

Options:

positional arguments:
  command
    clean        Remove unused packages and caches.
    config       Modify configuration values in .condarc. This is modeled
                 after the git config command. Writes to the user .condarc
                 file (/Users/dmertz/.condarc) by default.
    create       Create a new conda environment from a list of specified
                 packages.
    help         Displays a list of available conda commands and their help
                 strings.
    info         Display information about current conda install.
    install      Installs a list of packages into a specified conda
                 environment.
    [... more commands ...]

optional arguments:
  -h, --help     Show this help message and exit.
  -V, --version  Show the conda version number and exit.


Instructions
100 XP

        Run a command to determine what version of conda you have installed.

Solution : 

conda --version
'''