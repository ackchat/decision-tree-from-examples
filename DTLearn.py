"""
Author - Ackchat Omar
Title - Decision-Tree learning from examples
Summary - 
	Implement a basic decision-tree learning algorithm which uses 'Information_Gain'
	as the basis of IMPORTANCE function.
"""

from typing import Dict # For type_specification of 'dict' type
import pdb # For Debugging

# Class Node :
class Node:
	def __init__(self, atr_name : str, output = None): # atr_name -> attribute_name; children -> branch_map( atr_value --> child_node )
		self.atr_name = atr_name
		self.children = {}
		self.output = output
	
	def addChild(self, atr_value : str, child_node):
		self.children[atr_value] = child_node

	def print(self, baseIndentation = 0):
		if self.output is not None:
			print(f"|-{self.output} : Leaf-Node")
		else:
			print(f"|-{self.atr_name}")
			# Iterate over branches and do recursive calls :
			maxValueLen = max([len(key) for key in self.children.keys()])
			for key, value in self.children.items():
				outputText = f"{' ' * (baseIndentation + 4)}{key:<{maxValueLen}}  "
				print(outputText, end='')
				value.print(baseIndentation + len(outputText))
				
				

# Read CSV file :
import pandas as pd

data = pd.read_csv("restaurant.csv", header=None) # No column-header
data = data.map(lambda x: x.strip())
data.columns = ['Alt', 'Bar', 'Fir', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait']

"""
Take a look at the data :
print(data.describe());
data.info()		
print(data.shape)
"""
print(data)

# ENTROPY : For Binary Variables 
import math
def entropy(p, n):
	if p == 0 and n == 0:
		raise ValueError("Both p and n cannot be 0")
	prob_p = p/float(p+n) 
	if (prob_p == 1.0 or prob_p == 0.0):
		return 0.0
	return -(prob_p*math.log2(prob_p) + (1-prob_p)*(math.log2(1-prob_p)))

"""
# -- PASSED --
# Test Entropy :
# 1 : Invalid Input Values
print(entropy(0, 0)) # Gives Error 
# 2 : Try (p,n) values given in the book
print(entropy(1, 1) == 1)
print(abs(entropy(0.99, 0.01) - 0.08) < 0.001)
"""

# IMPORTANCE :
def infoGain(atr_name, target_atr_name, examples):
	# Calculate p = no. of +ve and n = no. of -ve :
	p = len(examples[examples[target_atr_name] == 'Yes'])
	n = len(examples) - p

	# Calculate Remainder(atr_name) :
	remainder : float = 0.0
	atr_values = set(examples[atr_name])
	for value in atr_values :
		subset = examples[examples[atr_name] == value]
		pk = len(subset[subset['WillWait'] == 'Yes'])
		if (len(subset) == 0):
			continue
		
		remainder += (len(subset)/float(len(examples))) * entropy(pk, len(subset)-pk)
	# Calculate Initial Entropy:
	initialE = entropy(p,n)
	return initialE - remainder

"""
# -- PASSED --
# Test infoGain :
# 1 : from 'restaurant examples in the book' {Gain(Patrons), Gain(Type)}
print(infoGain('Pat', 'WillWait', data)) # ~ 0.541
print(infoGain('Type', 'WillWait', data)) # 0 
"""


# PLURALITY-VALUE :
import random
"""
	This function takes in 2 arguments :
		1. target_atr_name - Column name of target-attribute
		2. examples - dataFrame which holds the examples
	This returns the most common 'value' of __target_atr_name__ from __examples__ dataset,
	breaking ties randomly.
"""
def pluralityValue(target_atr_name, examples):
	# Filter out highest count in 'target_atr_name' :
	target_values = set(examples[target_atr_name])
	maxCount = 0
	mostCommonValue = None
	for value in target_values:
		currCount = list(examples[target_atr_name]).count(value)
		if (currCount > maxCount):
			maxCount = currCount
			mostCommonValue = value
		elif (currCount == maxCount and mostCommonValue != value):
			mostCommonValue = random.choice([mostCommonValue, value])
			
	return mostCommonValue			

"""
# -- PASSED --
# test pluralityValue : CHECKED ALREADY
"""

# Decision-Tree Learning :
def dtLearn(examples, attributes, target_atr_name, parent_examples):
	# Check for trivial classifications :
	if (len(examples) == 0): # Examples is Empty
		return Node(None, pluralityValue(target_atr_name, parent_examples))
	elif (len(set(examples[target_atr_name])) == 1): # All Examples share same Classification
		#pdb.set_trace()
		return Node(None, examples[target_atr_name].iloc[0])
	elif (len(attributes) == 0): # Attributes is Empty
		return Node(None, pluralityValue(target_atr_name, examples))
	else:
		infoGains = [infoGain(atr_name, target_atr_name, examples) for atr_name in attributes] # Calculating the Information-Gains
		maxGainIdx = infoGains.index(max(infoGains))
		optimalAtrName = attributes[maxGainIdx] # Finding the Attribute with Maximum Information-Gain
		root = Node(optimalAtrName) # Root Node of a New Decision-Tree
		attributes.remove(optimalAtrName)
		for val in set(data[optimalAtrName]):
			exampleSubset = examples[examples[optimalAtrName] == val]
			child = dtLearn(exampleSubset, attributes, target_atr_name, examples)
			root.addChild(val, child)
		return root


# Running the Algorithm and Printing the Decision-Tree :
dtLearn(data, list(data.columns[:-1]), 'WillWait', data).print()
