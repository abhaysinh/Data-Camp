'''
Find dependencies for a package version?

The conda search package_name --info command reports a variety of details about a specific package. The syntax for specifying just one version is a little bit complex, but prefix notation is allowed here (just as you would with conda install).

For example, running conda search cytoolz=0.8.2 --info will report on all available package versions. As this package has been built for a variety of Python versions, a number of packages will be reported on. You can narrow your query further with, e.g.:

$ conda search cytoolz=0.8.2=py36_0 --info

cytoolz 0.8.2 py36_0
<hr />-----------------
file name   : cytoolz-0.8.2-py36_0.tar.bz2
name        : cytoolz
version     : 0.8.2
build string: py36_0
build number: 0
channel     : https://repo.anaconda.com/pkgs/free/osx-64
size        : 352 KB
arch        : x86_64
constrains  : ()
date        : 2016-12-23
license     : BSD
md5         : cd6068b2389b1596147cc7218f0438fd
platform    : darwin
subdir      : osx-64
url         : https://repo.anaconda.com/pkgs/free/osx-64/cytoolz-0.8.2-py36_0.tar.bz2
dependencies:
    python 3.6*
    toolz >=0.8.0

You may use the * wildcard within the match pattern. This is often useful to match 'foo=1.2.3=py36*' because recent builds have attached the hash of the build at the end of the Python version string, making the exact match unpredictable.

Determine the dependencies of the package numpy 1.13.1 with Python 3.6.0 on your current platform.

Instructions
50 XP

Possible Answers

    libgcc-ng >=7.2.0, libgfortran-ng >=7.2.0,<8.0a0, python >=2.7,<2.8.0a0, mkl >=2019.0.0, and blas * mkl
    libgcc-ng >=7.3.0, libgfortran-ng >=7.2.0, python >=3.5, mkl >=2018.0.0, and blas * mkl
    libgcc-ng >=7.2.0, libgfortran-ng >=7.2.0,<8.0a0, python >=2.7,<2.8.0a0, mkl >=2019.0.0,<2019.0a0, and blas * mkl
    libgcc-ng >=7.2.0, libgfortran-ng >=7.2.0,<8.0a0, python >=3.6,<3.7.0a0, mkl >=2018.0.0,<2019.0a0, and blas * mkl
    libgcc-ng >=7.3.0, libgfortran-ng >=7.2.0,<8.0a0, python >=3.6,<3.7.0a0, mkl >=2019.0.0,<2019.0a0, and blas * mkl


Answer :    libgcc-ng >=7.2.0, libgfortran-ng >=7.2.0,<8.0a0, python >=3.6,<3.7.0a0, mkl >=2018.0.0,<2019.0a0, and blas * mkl
'''