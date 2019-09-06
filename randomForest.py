import pandas as pd
import numpy as np
from fileReader import trainData
from sklearn.model_selection import train_test_split
from featureExtractor import extractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
import atexit


td = trainData(threshold=50)
label,raw = td.getLabelsAndrawData()

#process data
ext = extractor()
atexit.register(ext.saveCacheFiles)
ext.loadCacheFile()
ext.highFrequencyTokens(label, raw)
ext.extractEmoji(raw)
ext.extractHashTags(raw)
data = ext.batchProduceFixFeatureVec(raw)
ext.saveCacheFiles()
td.unloadData()

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)

#gnb = GaussianNB()
#gnb.fit(X_train, y_train)
#y_pred = gnb.predict(X_test)

# train model
regressor = RandomForestClassifier(n_estimators=20, criterion='entropy', verbose=10, n_jobs=2)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# print output
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

