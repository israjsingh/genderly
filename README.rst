=========
Genderly 
=========

============
Description
============

A library to classify an indian name as Male or Female. Mainly focused around making the user onboarding smooth for eCommerce merchants by pre-filling the gender for their users during a signup.

================
Quick Start
================

**Installation** ::

	pip install genderly

**Usage**

The library comes with a pre trained model which can be used as follows ::

	import genderly
	genderly.decide_gender(['raj'])

The function *decide_gender()* taked *Pandas Series object* and *lists* as input. 

In case one wants to rerun the training ::

	import genderly
	genderly.start_training()

The model has been trained on the basis of open Indian names database. One can add more data to the csv file which can be found at *genderly/data/list_NameGender.csv*



