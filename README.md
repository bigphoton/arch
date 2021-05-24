# arch

System architecture tools for big quantum photonics.

## conventions

 * Tabs are used for indentation throughout. (4 spaces per tab)
 * Initial and date your FIXME and TODO notes

## philosophy

`arch` should be
 * everything to everyone in quantum photonics
 * clear, concise, powerful

## Dependencies
Python 3.9 is required.

The non built-in packages required for this project are found in requirements.txt and can
are installed as usual with `pip install -r requirements.txt`

There is currently a bug which requires the use of sympy 1.7, rather than the most recent 1.8.
 
## open problems

How should we handle time?
 - Clocked
 - Time-independent
 	- works for some optics, not others
 - Synchronous
 	- works for everything
 	- time keeping tricky
 - Continuous time
 	- most rigorous


## contributors

### University of Bristol
JW Silverstone, S Currie, DA Quintero Dominguez, M Radulovic, D Roberts, LM Rosenfeld
