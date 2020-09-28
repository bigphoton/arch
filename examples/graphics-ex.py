"""
Example graphics.
"""

# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys
sys.path.insert(0,'..')

from arch.vis import generic, graphic


print("Hello world")

g = generic.generic_box()

graphic.graphic.wait_until_close()