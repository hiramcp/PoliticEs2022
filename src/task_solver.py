import os, csv, fasttext, math, nltk, time, pandas as pd, numpy as np

from os import path
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

dir = path.dirname(__file__)

def build_data(aInput, dfClass, ClassName, bEncondeY):
    X=[]
    Y=[]
    

    X = aInput.tolist()
    Y = dfClass[ClassName].tolist()
    le = preprocessing.LabelEncoder()

    if bEncondeY:
        Y = le.fit_transform(Y)


    return X, Y, le

#Main

#Preparing data files
#Emb [gender, profession, ideology_binary, ideology_multiclass]
fiNaTrain = "../data/emb_train.npy"
fiNaKlassTrain = "../data/df_class_train.csv"
fiTrain = path.abspath(path.join(dir,fiNaTrain))
fiKlassTrain= path.abspath(path.join(dir,fiNaKlassTrain))

#Load data
aTrain = np.load(fiTrain)
dfTrain = pd.read_csv(fiKlassTrain)

# 12-dim stacked vector
# Uncomment following block to use a 12-dim representation
# Comment the 3-dim block
# aTrainGen = aTrain
# aTrainProf = aTrain
# aTrainIdeoB = aTrain
# aTrainIdeoM = aTrain

#3-dim stacked vector
#Uncomment following block to use a 3-dim representation
# Comment the 12-dim block
aTrainGen = aTrain[:,0:3]
aTrainProf = aTrain[:,3:6]
aTrainIdeoB = aTrain[:,6:9]
aTrainIdeoM = aTrain[:,9:12]

bNormalize = True

# Gender

X_train, Y_train, le_train_gender = build_data(aTrainGen, dfTrain, "gender",True)

if bNormalize:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.fit_transform(X_dev)
    
#Training
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

mgb = GradientBoostingClassifier()

grid = GridSearchCV(mgb, param_grid, refit = True, verbose = 1, cv=cv)
grid.fit(X_train, Y_train)

mgb_best = grid.best_estimator_
mgb_best.fit(X_train, Y_train)

print('GradientBoosting CV Gender')
scores = cross_val_score(mgb_best, X_train, Y_train, scoring='f1_macro', cv=cv, n_jobs=-1)
print('f1_macro: %.3f desv. std:%.3f' % (scores.mean(), scores.std()))

#Profession
X_train, Y_train, le_train_prof = build_data(aTrainProf, dfTrain, "profession",True)

if bNormalize:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.fit_transform(X_dev)
    
#Training
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

mgb = GradientBoostingClassifier()

grid = GridSearchCV(mgb, param_grid, refit = True, verbose = 1, cv=cv)
grid.fit(X_train, Y_train)

mgb_best = grid.best_estimator_
mgb_best.fit(X_train, Y_train)

print('GradientBoosting CV Profession')
scores = cross_val_score(mgb_best, X_train, Y_train, scoring='f1_macro', cv=cv, n_jobs=-1)
print('f1_macro: %.3f desv. std:%.3f' % (scores.mean(), scores.std()))

#ideology_binary
X_train, Y_train, le_train_ideob = build_data(aTrainIdeoB, dfTrain, "ideology_binary",True)

if bNormalize:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.fit_transform(X_dev)
    
#Training
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

mgb = GradientBoostingClassifier()

grid = GridSearchCV(mgb, param_grid, refit = True, verbose = 1, cv=cv)
grid.fit(X_train, Y_train)

mgb_best = grid.best_estimator_
mgb_best.fit(X_train, Y_train)

print('GradientBoosting CV ideology_binary')
scores = cross_val_score(mgb_best, X_train, Y_train, scoring='f1_macro', cv=cv, n_jobs=-1)
print('f1_macro: %.3f desv. std:%.3f' % (scores.mean(), scores.std()))

#ideology_multiclass

X_train, Y_train, le_train_ideom = build_data(aTrainIdeoM, dfTrain, "ideology_multiclass",True)

if bNormalize:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.fit_transform(X_dev)
    
#Training
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

mgb = GradientBoostingClassifier()

grid = GridSearchCV(mgb, param_grid, refit = True, verbose = 1, cv=cv)
grid.fit(X_train, Y_train)

mgb_best = grid.best_estimator_
mgb_best.fit(X_train, Y_train)

print('GradientBoosting CV ideology_multiclass')
scores = cross_val_score(mgb_best, X_train, Y_train, scoring='f1_macro', cv=cv, n_jobs=-1)
print('f1_macro: %.3f desv. std:%.3f' % (scores.mean(), scores.std()))

