# Arch Documentation

This documentation describes the underlying mathematics and software implementation of the `arch` quantum photonics simulation software.

## Motivation

* <Include project motivation here>

## Structure

1. [Overview](https://github.com/BigPhoton/arch/docs/overview/README.md)
2. [Device Models Theory and Examples](https://github.com/BigPhoton/arch/docs/models/README.md)
3. [Simulators Theory and Application](https://github.com/BigPhoton/arch/docs/simulation/README.md)
4. Analyzer tools

## Get Started

Make sure to run these commandsbefore committing these example Jupyter notebook to `arch`.

```bash
pip3 install nbdime
nbdime config-git --enable --global`
```

Consider [making a branch and pull request when merging to the master branch](https://gist.github.com/vlandham/3b2b79c40bc7353ae95a), but should not be necessary - just cleaner and safer when merging if conflicts arise.

## Stylistic Guidelines¶
We're aiming to follow best coding style practices like any other large open-source Python project. We're aiming to do somthing like the [the NumPy Style guide](https://numpy.org/doc/stable/dev/index.html#stylistic-guidelines):

[Set up your editor](https://blog.jetbrains.com/pycharm/2013/02/long-awaited-pep-8-checks-on-the-fly-improved-doctest-support-and-more-in-pycharm-2-7/) to follow [PEP 8](https://realpython.com/python-pep8/) (remove trailing white space, no tabs, etc.). Using PyCharm will lift the weight off your shoulders on writing working and nice code. Other useful tools that help writing nice code are: [black](https://github.com/psf/black) and [flake8](https://pypi.org/project/flake8/).
