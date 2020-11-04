"""
Example graphics.
"""

import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from arch.vis import generic, graphic


print("Hello world")

g = generic.generic_box()

graphic.graphic.wait_until_close()