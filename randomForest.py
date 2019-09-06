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

td = trainData(threshold=50)
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
#ext.saveCacheFiles()
ext.unloadExt()
del ext
del raw

print(getsizeof(data))
print("clean trash...")

#X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)



#gnb = GaussianNB()
#gnb.fit(X_train, y_train)
#y_pred = gnb.predict(X_test)

# train model
regressor = RandomForestClassifier(n_estimators=20, criterion='entropy', verbose=10, n_jobs=2)
regressor.fit(data, label)

pickle.dump(regressor, open("rf.model", 'wb'))
data = None
label = None
y_pred = regressor.predict(tdata)

fileReader.writeToCsv(y_pred)
print("finished!!!!")
# print outpu
