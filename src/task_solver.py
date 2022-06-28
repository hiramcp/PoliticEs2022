#from juliacall import Main as jl
#jl.seval("using NPZ")

#emb_train = jl.NPZ.npzread("C:\\HCP\\Personal\\Doctorado\\3er Semestre\\iberLef 22\\Datos\\emb_train.npy")
#emb_dev = jl.NPZ.npzread("C:\\HCP\\Personal\\Doctorado\\3er Semestre\\iberLef 22\\Datos\\emb_dev.npy")


#from msilib.schema import Class
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
started = time.time()
print("Inicio Tiempo seg: " , time.strftime("%H:%M:%S", time.localtime()))

#Preparing data files
#Emb [gender, profession, ideology_binary, ideology_multiclass]
fiNaTrain = "../../Datos/emb_train.npy"
fiNaDev = "../../Datos/emb_dev.npy"

fiNaKlassTrain = "../../Datos/df_class_train.csv"
fiNaKlassDev = "../../Datos/df_class_dev.csv"

fiTrain = path.abspath(path.join(dir,fiNaTrain))
fiDev = path.abspath(path.join(dir,fiNaDev))

fiKlassTrain= path.abspath(path.join(dir,fiNaKlassTrain))
fiKlassDev= path.abspath(path.join(dir,fiNaKlassDev))

#Load data
aTrain = np.load(fiTrain)
aDev = np.load(fiDev)

dfTrain = pd.read_csv(fiKlassTrain)
dfDev = pd.read_csv(fiKlassDev)

# aTrainGen = aTrain
# aTrainProf = aTrain
# aTrainIdeoB = aTrain
# aTrainIdeoM = aTrain

# aDevGen = aDev
# aDevProf = aDev
# aDevIdeoB = aDev
# aDevIdeoM = aDev

aTrainGen = aTrain[:,0:3]
aTrainProf = aTrain[:,3:6]
aTrainIdeoB = aTrain[:,6:9]
aTrainIdeoM = aTrain[:,9:12]

aDevGen = aDev[:,0:3]
aDevProf = aDev[:,3:6]
aDevIdeoB = aDev[:,6:9]
aDevIdeoM = aDev[:,9:12]

bNormalize = True

#Genero
print("Iniciando Genero")

X_train, Y_train, le_train_gender = build_data(aTrainGen, dfTrain, "gender",True)
X_dev, Y_dev, le_dev_gender = build_data(aDevGen, dfDev, "gender", True)

# X_train, Y_train, le_train_gender = build_data(aTrain, dfTrain, "gender",True)
# X_dev, Y_dev, le_dev_gender = build_data(aDev, dfDev, "gender", True)

if bNormalize:

    #O se normaliza o se escala, no ambos
    # from sklearn import preprocessing
    # twVectorNorm_train = preprocessing.normalize(twVector_train)
    # twVectorNorm_test = preprocessing.normalize(twVector_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.fit_transform(X_dev)
    
#Training
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
#n_estimators = [3000]
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

mgb = GradientBoostingClassifier()

grid = GridSearchCV(mgb, param_grid, refit = True, verbose = 1, cv=cv)
grid.fit(X_train, Y_train)

print(grid.best_params_)
print(grid.best_estimator_)

mgb_best = grid.best_estimator_
mgb_best.fit(X_train, Y_train)

print('GradientBoosting CV Gender')
scores = cross_val_score(mgb_best, X_train, Y_train, scoring='f1_macro', cv=cv, n_jobs=-1)
print('f1_macro: %.3f desv. std:%.3f' % (scores.mean(), scores.std()))

mgb_predict_test = mgb_best.predict(X_dev)

# print('GradientBoosting Test Gender')
# print("Acc M =", accuracy_score(Y_dev, mgb_predict_test))
# print("F1 M =", f1_score(Y_dev, mgb_predict_test, average='macro'))
# print("micro F1 M =", f1_score(Y_dev, mgb_predict_test, average='micro'))

print("Tiempo seg Genero: " , str(time.time() - started))

#Profession
print("Iniciando profesion")

X_train, Y_train, le_train_prof = build_data(aTrainProf, dfTrain, "profession",True)
X_dev, Y_dev, le_dev_prof = build_data(aDevProf, dfDev, "profession", True)

# X_train, Y_train, le_train_prof = build_data(aTrain, dfTrain, "profession",True)
# X_dev, Y_dev, le_dev_prof = build_data(aDev, dfDev, "profession", True)

if bNormalize:
    #O se normaliza o se escala, no ambos
    # from sklearn import preprocessing
    # twVectorNorm_train = preprocessing.normalize(twVector_train)
    # twVectorNorm_test = preprocessing.normalize(twVector_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.fit_transform(X_dev)
    
#Training
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
#n_estimators = [3000]
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

mgb = GradientBoostingClassifier()

grid = GridSearchCV(mgb, param_grid, refit = True, verbose = 1, cv=cv)
grid.fit(X_train, Y_train)

print(grid.best_params_)
print(grid.best_estimator_)

mgb_best = grid.best_estimator_
mgb_best.fit(X_train, Y_train)

print('GradientBoosting CV Profession')
scores = cross_val_score(mgb_best, X_train, Y_train, scoring='f1_macro', cv=cv, n_jobs=-1)
print('f1_macro: %.3f desv. std:%.3f' % (scores.mean(), scores.std()))

mgb_predict_test = mgb_best.predict(X_dev)

# print('GradientBoosting Test Profession')
# print("Acc M =", accuracy_score(Y_dev, mgb_predict_test))
# print("F1 M =", f1_score(Y_dev, mgb_predict_test, average='macro'))
# print("micro F1 M =", f1_score(Y_dev, mgb_predict_test, average='micro'))

print("Tiempo seg Profesion: " , str(time.time() - started))

#ideology_binary
print("Iniciando Ideologia binaria")

X_train, Y_train, le_train_ideob = build_data(aTrainIdeoB, dfTrain, "ideology_binary",True)
X_dev, Y_dev, le_dev_ideob = build_data(aDevIdeoB, dfDev, "ideology_binary", True)

# X_train, Y_train, le_train_ideob = build_data(aTrain, dfTrain, "ideology_binary",True)
# X_dev, Y_dev, le_dev_ideob = build_data(aDev, dfDev, "ideology_binary", True)

if bNormalize:
    #O se normaliza o se escala, no ambos
    # from sklearn import preprocessing
    # twVectorNorm_train = preprocessing.normalize(twVector_train)
    # twVectorNorm_test = preprocessing.normalize(twVector_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.fit_transform(X_dev)
    
#Training
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
#n_estimators = [3000]
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

mgb = GradientBoostingClassifier()

grid = GridSearchCV(mgb, param_grid, refit = True, verbose = 1, cv=cv)
grid.fit(X_train, Y_train)

print(grid.best_params_)
print(grid.best_estimator_)

mgb_best = grid.best_estimator_
mgb_best.fit(X_train, Y_train)

print('GradientBoosting CV ideology_binary')
scores = cross_val_score(mgb_best, X_train, Y_train, scoring='f1_macro', cv=cv, n_jobs=-1)
print('f1_macro: %.3f desv. std:%.3f' % (scores.mean(), scores.std()))

mgb_predict_test = mgb_best.predict(X_dev)

# print('GradientBoosting Test ideology_binary')
# print("Acc M =", accuracy_score(Y_dev, mgb_predict_test))
# print("F1 M =", f1_score(Y_dev, mgb_predict_test, average='macro'))
# print("micro F1 M =", f1_score(Y_dev, mgb_predict_test, average='micro'))

print("Tiempo seg Ideologia binaria: " , str(time.time() - started))
#ideology_multiclass

print("Iniciando ideologia multiclase")

X_train, Y_train, le_train_ideom = build_data(aTrainIdeoM, dfTrain, "ideology_multiclass",True)
X_dev, Y_dev,le_dev_ideom = build_data(aDevIdeoM, dfDev, "ideology_multiclass", True)

# X_train, Y_train, le_train_ideom = build_data(aTrain, dfTrain, "ideology_multiclass",True)
# X_dev, Y_dev,le_dev_ideom = build_data(aDev, dfDev, "ideology_multiclass", True)

if bNormalize:
    #O se normaliza o se escala, no ambos
    # from sklearn import preprocessing
    # twVectorNorm_train = preprocessing.normalize(twVector_train)
    # twVectorNorm_test = preprocessing.normalize(twVector_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.fit_transform(X_dev)
    
#Training
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
#n_estimators = [3000]
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

mgb = GradientBoostingClassifier()

grid = GridSearchCV(mgb, param_grid, refit = True, verbose = 1, cv=cv)
grid.fit(X_train, Y_train)

print(grid.best_params_)
print(grid.best_estimator_)

mgb_best = grid.best_estimator_
mgb_best.fit(X_train, Y_train)

print('GradientBoosting CV ideology_multiclass')
scores = cross_val_score(mgb_best, X_train, Y_train, scoring='f1_macro', cv=cv, n_jobs=-1)
print('f1_macro: %.3f desv. std:%.3f' % (scores.mean(), scores.std()))

mgb_predict_test = mgb_best.predict(X_dev)

# print('GradientBoosting Test ideology_multiclass')
# print("Acc M =", accuracy_score(Y_dev, mgb_predict_test))
# print("F1 M =", f1_score(Y_dev, mgb_predict_test, average='macro'))
# print("micro F1 M =", f1_score(Y_dev, mgb_predict_test, average='micro'))

print("Tiempo seg ideologia multiclase: " , str(time.time() - started))

ended = time.time() - started
print("Fin del proceso {Tiempo seg: " , str(ended))

