'''
Created on 28 Sep 2020

@author: lawrencerosenfeld
'''

import unittest
from os.path import dirname

def main():
	# Look for all unittests.TestCases in this folder, with a prefix of *_test.py
	path = dirname(__file__)
	suite = unittest.TestLoader().discover(path)

	# Run
	unittest.TextTestRunner(verbosity=1).run(suite)
	
	return

if __name__ == '__main__':
	main()