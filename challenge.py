import pandas as pd
import numpy as np 
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#read data
dataframe = pd.read_csv('challenge_dataset.txt')
x_values = np.array(dataframe[['X']])
y_values = np.array(dataframe[['Y']])
#split data in training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.4, random_state=0)


linear_clf = linear_model.LinearRegression()
svm_linear_clf = svm.SVR(kernel= 'linear', C= 1e3) # defining the support vector regression models
svm_poly_clf = svm.SVR(kernel='poly', C= 1e3)
#train classifier
linear_clf.fit(X_train, y_train)
svm_linear_clf.fit(X_train, y_train.ravel())
svm_poly_clf.fit(X_train, y_train.ravel())


print 'Score Poly SVM (on test set): ', svm_poly_clf.score(X_test, y_test)
print 'Score Poly SVM (on entire set): ', svm_poly_clf.score(x_values, y_values)
print 'Mean Absolute Error: ', mean_absolute_error(y_values, linear_clf.predict(x_values))
plt.scatter(x_values, y_values, color='black', label='Data')
plt.plot(x_values, svm_poly_clf.predict(x_values), color='red', label='Poly SVM Model')
plt.legend()
plt.show()
