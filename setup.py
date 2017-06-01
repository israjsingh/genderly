from setuptools import setup
import os
setup(name='genderly',
      version='0.105',
      description='A library to classify an indian name as Male or Female. Mainly focused around making the user onboarding smooth for eCommerce merchants by pre-filling the gender for their users during a signup.',
      url='https://github.com/rajdeltarobo/genderly',
      author='Raj Vardhan Singh',
      author_email='is.rajvardhan@gmail.com',
      license='MIT',
      packages=['genderly'],
      package_data = {
        'genderly/data/': [os.path.join('genderly/data','list_NameGender.csv'),os.path.join('genderly/data','list_NameGender.json')],
        'genderly/model/': [os.path.join('genderly/model','*.pkl'),os.path.join('genderly/model','*.npy')]
    },
      zip_safe=False)