import unittest
import importlib
import os,sys

__all__ = ['test_installation']

class test_installation(unittest.TestCase):
	'''
	Test correct installation and importing of packages in requirements.txt
	'''

	def test_import_requirements(self):
		# Load the module strings in requirements.txt as a list 
		fn = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
		with open(fn, 'r') as f:
			module_names = f.readlines()
		
		# Loop through modules and keep track of which ones fail importing
		errors = []
		for m in module_names:
			try:
				importlib.import_module(m)
			except:
				errors.append(str(sys.exc_info()[:2][1]))
		
		if errors:
			raise ImportError(', '.join(errors))
		
		
		return

if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main()