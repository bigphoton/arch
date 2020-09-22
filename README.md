# arch
System architecture tools for big quantum photonics.

## conventions

 * Tabs are used for indentation throughout. (4 spaces per tab)
 * Lines are wrapped hard at 120 characters length.
 * Classes are in lower-case with underscores, `like_this`.
 * Boolean state flags should be named as `is_*`: `is_hot`, `is_real`, `is_friendly`, etc.
 * Global constants are in upper-case, `LIKE_THIS`.
 * Keep names short
 * Give and take credit where it's due
 * Initial and date your FIXME and TODO notes

## philosophy

`arch` should be
 * everything to everyone in quantum photonics
 * as close to perfect as possible: clear, concise, powerful
 

## graphics
 
If desired, graphics are currently implemented using the built-in `turtle` module and `tkinter`. Export to PostScript is
possible, but export to other formats is currently not supported.
 
 
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

