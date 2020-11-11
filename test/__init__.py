# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))