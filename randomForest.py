import pandas as pd
import numpy as np
from fileReader import trainData, testData
import fileReader
from sklearn.model_selection import train_test_split
from featureExtractor import extractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sys import getsizeof
import pickle
import atexit
import gc
import ctypes


#def malloc_trim():
    #ctypes.CDLL('libc.so.6').malloc_trim(0)

def scope():
    td = trainData(threshold=20)
    testd = testData()
    label,raw = td.getLabelsAndrawData()

    #process data
    ext = extractor()

    ext.loadCacheFile()
    ext.highFrequencyTokens(label, raw)
    ext.extractEmoji(raw)
    ext.extractHashTags(raw)
    data = ext.batchProduceFixFeatureVec(raw)
    tdata = ext.batchProduceFixFeatureVec(testd.getAllTweets())
    td.unloadData()
    ext.saveCacheFiles()
    #ext.unloadExt()
    del ext
    del raw


    print(getsizeof(data))
    print("clean trash...")

    #malloc_trim()


    return label, data, tdata

#X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)



#gnb = GaussianNB()
#gnb.fit(X_train, y_train)
#y_pred = gnb.predict(X_test)

label, data, tdata = scope()

# train model
regressor = RandomForestClassifier(n_estimators=30, criterion='entropy', verbose=10, n_jobs=8)
regressor.fit(data, label)

pickle.dump(regressor, open("rf.model", 'wb'))
data = None
label = None
y_pred = regressor.predict(tdata)

fileReader.writeToCsv(y_pred)
print("finished!!!!")
# print outpu
