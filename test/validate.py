import unittest
from os.path import dirname

def main():
	'''
	This script runs all unit tests associated with the arch project 
	'''
	
	# Look for all unittests.TestCases in this folder, with a prefix of test_*.py
	path = dirname(__file__)
	suite = unittest.TestLoader().discover(path)

	# Run
	unittest.TextTestRunner(verbosity=1).run(suite)
	
	return

if __name__ == '__main__':
	main()