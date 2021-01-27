#!/usr/bin/env python
import setuptools

with open("README_RAW.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arch",
    version="0.0.1",
    description="Arch - Classical adn quantum photonic systems architecture tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BigPhoton/arch",
    author="TODO", #TODO authors.txt
    author_email="TODO", #TODO bigphoton lab email, necessary for PyPi
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
