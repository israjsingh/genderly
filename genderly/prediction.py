import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')

def syllables(word):
	count = 0
	vowels = 'aeiouy'
	#print word
	word = word.strip(".:;?!")
	if word[0] in vowels:
	    count +=1
	for index in range(1,len(word)):
	    if word[index] in vowels and word[index-1] not in vowels:
	        count +=1
	if word.endswith('e'):
	    count -= 1
	if word.endswith('le'):
	    count+=1
	if count == 0:
	    count +=1
	return count

def feature_engineering(FirstName):
	if isinstance(FirstName,pd.Series) is True:
		df=create_features(FirstName)
		return df
	elif isinstance(FirstName,list) is True:
		FirstName=pd.Series(FirstName)
		df=create_features(FirstName)
		return df
	else:
		return('Expected Dataframe or list')

def create_features(FirstName):

	allowed_NLast1=['a','d','e','g','h','i','j','k','l','m','n','o','p','r','s','t','u','v','y','z','others']

	allowed_NLast2=['aa','ah','ai','aj','ak','al','am','an','ar','as','at','av','ay','ba','da','dh',\
	                'di','du','ee','el','em','en','ep','er','et','ev','ey','ga','gi','ha','hi','hu',\
	                'hy','ia','ik','il','in','ip','ir','is','it','iv','ja','ji','ju','ka','kh','ki',\
	                'la','li','lu','ma','mi','na','ne','ni','nt','nu','ny','oj','ol','on','op','or',\
	                'ot','pa','pi','ra','re','ri','ru','sa','sh','si','su','ta','th','ti','tu','ty',\
	                'ul','um','un','ur','va','vi','ya','yi','yu','others']

	allowed_NLast3=['aam','aan','ada','aja','aka','ala','ali','ama','ami','ana','ang','ani','ant',\
	                'anu','ara','ari','asi','ata','ath','ati','ava','avi','aya','ayi','bha','bir',\
	                'chi','der','dev','dha','dhi','dhu','dra','dya','eel','een','eep','eer','eet',\
	                'ena','era','esh','eva','evi','eya','hal','ham','han','har','hay','hka','hma',\
	                'hmi','hna','hra','hri','hti','hvi','hya','ija','ika','ila','ima','ina','ini',\
	                'ira','ish','ita','ith','iti','iya','jal','jay','jit','kar','kha','khi','kta',\
	                'lya','mal','may','mya','nal','nam','nay','nda','ndu','nee','nga','ngi','nta',\
	                'nti','nvi','nya','odh','oop','oti','pal','raj','ram','ran','rat','ree','rin',\
	                'rit','rna','rti','rya','sha','shi','shu','sri','sti','tal','tha','thi','thy',\
	                'tra','tri','tya','uja','ukh','ula','una','uri','uta','uti','van','vya','yam',\
	                'yan','others']

	other_columns=['NLast1','NLast2','NLast3','LastVowel','Length','CountVowels','LastVowelCount',\
	               'Last2VowelCount','Last3VowelCount','SyllablesCount','VowelsPercentage']

	NLastFeatures=['NLast1_'+ x for x in allowed_NLast1]
	NLast2Features=['NLast2_'+x for x in allowed_NLast2]
	NLast3Features=['NLast3_'+x for x in allowed_NLast3]
	all_columns=NLastFeatures+NLast2Features+NLast3Features+other_columns
	df=pd.DataFrame(columns=all_columns)
	df.ix[:,'FirstName']=FirstName
	df.ix[:,'FirstName']=FirstName
	df['NLast1']=df['FirstName'].str[-1]
	df['NLast2']=df['FirstName'].str[-2:]
	df['NLast3']=df['FirstName'].str[-3:]
	df.ix[df['NLast1'].isin(['a','e','i','o','u'])==True, 'LastVowel']=1
	df.ix[~df['NLast1'].isin(['a','e','i','o','u'])==True, 'LastVowel']=0
	df['Length']=df['FirstName'].str.len()
	df['CountVowels']=df['FirstName'].str.count(r'(a|e|i|o|u|y)')
	df['LastVowelCount']=df['NLast1'].str.count(r'(a|e|i|o|u|y)')
	df['Last2VowelCount']=df['NLast2'].str.count(r'(a|e|i|o|u|y)')
	df['Last3VowelCount']=df['NLast3'].str.count(r'(a|e|i|o|u|y)')
	df['SyllablesCount']=df['FirstName'].apply(syllables)
	df['VowelsPercentage']=(df['CountVowels']*100)/df['Length']

	df.ix[~df['NLast1'].isin(allowed_NLast1),'NLast1']='others'
	df.ix[~df['NLast2'].isin(allowed_NLast2),'NLast2']='others'
	df.ix[~df['NLast3'].isin(allowed_NLast3),'NLast3']='others'
	df[['NLast1_'+df['NLast1'].values[0]]] = 1
	df[['NLast2_'+df['NLast2'].values[0]]] = 1
	df[['NLast3_'+df['NLast3'].values[0]]] = 1
	return df

def decide_gender(name):
    genderPredictionModel = joblib.load('genderly/model/genderPredictionModel.pkl')
    data=feature_engineering(name)
    data.fillna(0,inplace=True)
    data.drop(['FirstName','NLast1','NLast2','NLast3'], axis=1, inplace=True)
    data=data.reset_index(drop=True)
    data.sort_index(axis=1,inplace=True)
    y_predicted=genderPredictionModel.predict(data)
    y_score=genderPredictionModel.predict_proba(data)
    print y_predicted,y_score
    return y_predicted,y_score

