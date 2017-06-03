=========
Genderly 
=========

**Description**

A library to classify an indian name as Male or Female. Mainly focused around making the user onboarding smooth for eCommerce merchants by pre-filling the gender for their users during a signup.


**Installation** ::

	pip install genderly

**Usage**

The library comes with a pre trained model which can be used as follows ::

	import genderly
	genderly.decide_gender(['raj'])

or ::
	
	import genderly
	genderly.decide_gender(df['first_name'])

Note : It is recommended that the first name being passed to the model is atleat 3 letter long. The model wouldnot work with last names or surnames. It has been trained on ly to work with first names. 
	

The function *decide_gender()* taked *Pandas Series object* and *lists* as input. 

In case one wants to rerun the training ::

	import genderly
	genderly.start_training()

The model has been trained on the basis of open Indian names database. One can add more data to the csv file which can be found at *genderly/data/list_NameGender.csv*

**Dependencies**
python 2.7
pandas
numpy
scikit-learn





