import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

PATH = './machinelearning-data/'

def labelstr2labelint(labelstr):
	labels= ['EnhancerInactive+PromoterInactive+Exon+Unknown', 'PromoterActive', 'EnhancerActive']
	return labels.index(labelstr)

X_train = np.loadtxt(PATH + 'GM12878_200bp_Data_3Cl_l2normalized_TrainSet.txt')
X_valid = np.loadtxt(PATH + 'GM12878_200bp_Data_3Cl_l2normalized_ValidSet.txt')
X_test = np.loadtxt(PATH + 'GM12878_200bp_Data_3Cl_l2normalized_TestSet.txt')

y_train = np.loadtxt(PATH + 'GM12878_200bp_Classes_3Cl_l2normalized_TrainSet.txt', converters = {0: labelstr2labelint})
y_valid = np.loadtxt(PATH + 'GM12878_200bp_Classes_3Cl_l2normalized_ValidSet.txt', converters = {0: labelstr2labelint})
y_test = np.loadtxt(PATH + 'GM12878_200bp_Classes_3Cl_l2normalized_TestSet.txt', converters = {0: labelstr2labelint})

X_train_valid = np.row_stack((X_train, X_valid))
y_train_valid = np.concatenate((y_train, y_valid))

tuned_parameters = [{'C': [0.1, 0.3, 1, 2, 3, 4, 5, 10, 30, 100]}]
clf = GridSearchCV(LogisticRegression(penalty='l1'), tuned_parameters, cv=3,
                   n_jobs=4, scoring='accuracy')
clf.fit(X_train_valid, y_train_valid)

print("Best parameters set found on development set:")
print(clf.best_params_)

clf_best = LogisticRegression(penalty='l1',C=clf.best_params_['C'])
clf_best.fit(X_train_valid,y_train_valid)

y_true, y_pred = y_test, clf_best.predict(X_test)
print('Accuracy:')
print(accuracy_score(y_true, y_pred))
print ('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))
print "Weights: "
print clf_best.coef_
