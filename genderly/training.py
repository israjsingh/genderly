import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
import warnings
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
warnings.filterwarnings('ignore')

def _datacleaning(data) :
    
    cleandataset = data[["FirstName","Gender"]].copy()
    cleandataset['FirstName']=cleandataset['FirstName'].str.strip('\r')
    cleandataset['FirstName']=cleandataset.FirstName.str.lower()
    cleandataset['Gender']=cleandataset.Gender.str.lower()
    cleandataset['FirstName']=cleandataset.FirstName.str.strip(' ')
    cleandataset=cleandataset.dropna(axis=0, how='any')
    return(cleandataset)

    
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

def _genderfeatures(df):
    df['NLast1']=df['FirstName'].str[-1]
    df['NLast2']=df['FirstName'].str[-2:]
    df['NLast3']=df['FirstName'].str[-3:]
#     df['NLast4']=df['FirstName'].str[-4:]
    df.ix[df['NLast1'].isin(['a','e','i','o','u'])==True, 'LastVowel']=1
    df.ix[~df['NLast1'].isin(['a','e','i','o','u'])==True, 'LastVowel']=0
    df['Length']=df['FirstName'].str.len()
    df['CountVowels']=df['FirstName'].str.count(r'(a|e|i|o|u|y)')
    df['LastVowelCount']=df['NLast1'].str.count(r'(a|e|i|o|u|y)')
    df['Last2VowelCount']=df['NLast2'].str.count(r'(a|e|i|o|u|y)')
    df['Last3VowelCount']=df['NLast3'].str.count(r'(a|e|i|o|u|y)')
    df['SyllablesCount']=df['FirstName'].apply(syllables)
#     df['VowelsPercentageLog']=np.log((df['CountVowels']*100)/df['Length'])
    df['VowelsPercentage']=(df['CountVowels']*100)/df['Length']

    
    allowed_NLast1=['a','d','e','g','h','i','j','k','l','m','n','o','p','r','s','t','u','v','y','z']
    
    allowed_NLast2=['aa','ah','ai','aj','ak','al','am','an','ar','as','at','av','ay','ba','da','dh',\
                    'di','du','ee','el','em','en','ep','er','et','ev','ey','ga','gi','ha','hi','hu',\
                    'hy','ia','ik','il','in','ip','ir','is','it','iv','ja','ji','ju','ka','kh','ki',\
                    'la','li','lu','ma','mi','na','ne','ni','nt','nu','ny','oj','ol','on','op','or',\
                    'ot','pa','pi','ra','re','ri','ru','sa','sh','si','su','ta','th','ti','tu','ty',\
                    'ul','um','un','ur','va','vi','ya','yi','yu']
    
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
                    'yan']
    
    df.ix[~df['NLast1'].isin(allowed_NLast1),'NLast1']='others'
    df.ix[~df['NLast2'].isin(allowed_NLast2),'NLast2']='others'
    df.ix[~df['NLast3'].isin(allowed_NLast3),'NLast3']='others'
    return df


def _testtrainsplit(rawdata):

    rand_num = np.random.rand(len(rawdata))<0.6
    train_raw = rawdata[rand_num]
    testValidate = rawdata[~rand_num]
    rand_test_num=np.random.rand(len(testValidate))<0.5
    test_raw = testValidate[rand_test_num]
    validate_raw = testValidate[~rand_test_num]
    return (train_raw,test_raw)

def _importraw() :
    dataset=pd.read_csv("genderly/data/list_NameGender.csv")
#   dataset= dataset[dataset.first_name.str.len()>3]
    return(dataset)

def _classifier_training(dataframe):
    
    sk = np.random.rand(len(dataframe)) < 0.8
    datatrain = dataframe[sk]
    datatest = dataframe[~sk]
    
    datatrain=datatrain.reindex_axis(sorted(datatrain.columns), axis=1)
    target = datatrain[['GenderDummy']].reset_index(drop=True)
    train = datatrain.drop('GenderDummy', axis=1).reset_index(drop=True)
    
    rf = RandomForestClassifier(n_estimators=20, 
                                        criterion='gini', 
                                        max_depth=20, 
                                        min_samples_split=5, 
                                        min_samples_leaf=5, 
                                        min_weight_fraction_leaf=0.0, 
                                        max_features=0.72, 
                                        max_leaf_nodes=None, 
                                        bootstrap=True, 
                                        oob_score=True, 
                                        n_jobs=-1, 
                                        random_state=876584, 
                                        verbose=0, 
                                        warm_start=True, 
                                        class_weight=None)
    
    rf.fit(train, target)
    scores = cross_val_score(rf, train, target['GenderDummy'], scoring='roc_auc',cv=4)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    feature_importance=[]
    for f in range(train.shape[1]):
        feature_importance.append(("feature %d (%f)" % (indices[f], importances[indices[f]])))

    trained_model=rf
#     print "CV Score : "
#     print scores
    feature_importance=feature_importance
#     print "Feature Importance"
#     print feature_importance
#     print "OOB Score"
#     print rf.oob_score_
    return rf,datatest

def start_training():
    df=_importraw()
    df_cleaned=_datacleaning(df)
    df_features=_genderfeatures(df_cleaned)
    dummy_features=pd.get_dummies(df_features[['NLast1','NLast2','NLast3']])
    df_dummy_features=pd.concat([df_cleaned, dummy_features], axis=1)

    df_dummy_features.ix[df_dummy_features['Gender']=="m","GenderDummy"]=0
    df_dummy_features.ix[df_dummy_features['Gender']=="f","GenderDummy"]=1
    df_dummy_features=df_dummy_features.drop(['FirstName','Gender','NLast1','NLast2','NLast3'], axis=1) 
    df_dummy_features.fillna(0, inplace=True)

    genderPredictionModel,testdata=_classifier_training(df_dummy_features)

    testdata.sort_index(axis=1,inplace=True)
    y_true=testdata[['GenderDummy']].reset_index(drop=True)
    X_testdata=testdata.drop('GenderDummy', axis=1).reset_index(drop=True)

    y_predicted=genderPredictionModel.predict(X_testdata)

    y_true=y_true.reset_index(drop=True)
    #y_predicted_1= [item[0] for item in y_predicted]
    precision, recall, thresholds = precision_recall_curve(y_true,y_predicted)
    avg_precision=average_precision_score(y_true,y_predicted)

    joblib.dump(genderPredictionModel, 'genderly/model/genderPredictionModel.pkl') 
    return("training complete")