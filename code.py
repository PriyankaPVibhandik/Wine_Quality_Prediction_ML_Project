""" Wine Quality Prediction:-

Using Classification Algorithms as follows:-

Decision Tree Classifier
K-NN Classifier
SVM Classifier

 """
#importing the libraries
import numpy as nm  
import matplotlib.pyplot as plt
import pandas as pd  
import seaborn as sns
from sklearn.model_selection import train_test_split

#importing the dataset
wine_dataset = pd.read_csv("/content/winequality-red.csv")
wine_dataset.head()
wine_dataset.shape
wine_dataset.columns
wine_dataset.info()

# checking for missing values
wine_dataset.isnull().sum()

#Descriptive data analysis and visualization:
wine_dataset.describe()
wine_dataset['quality'].value_counts()
# number of values for each quality
sns.catplot(x='quality', data = wine_dataset, kind = 'count')

#Data Preprocessing
#Extract the independent(input) and dependent(output) variables.
X = wine_dataset.drop('quality',axis=1)
Y = wine_dataset['quality'] # allocating the output to dependent variable which we want to predict
X = X.values
Y = Y.values
# X = wine_dataset.iloc[:,:-1].values 
print(X.shape, Y.shape)

#lable binarization
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(Y)

#Splitting the dataset into training and testing set.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 3)

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#Normalization of dataset
#Standardize the data (or perform Feature Scaling)
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

X_train = (X_train-X_mean)/X_std

X_test = (X_test-X_mean)/X_std

print(X_train.shape, X_test.shape)

#Feature scaling

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)
# print(X_train, X_test)


""" 
Training the model using classification algorithms.

Decision Tree Classifier
K-Nearest Neighbour(K-NN) Classifier
SVM Classifier
"""

#importing the libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, jaccard_score

#-----------------------------------------------------------------------------------------------
#1: Decision Tree Classifier

d_tree = DecisionTreeClassifier()
d_tree.fit(X_train,Y_train)

d_acc = accuracy_score(Y_test,d_tree.predict(X_test))

print("Train Set Accuracy:"+str(accuracy_score(Y_train,d_tree.predict(X_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(Y_test,d_tree.predict(X_test))*100))

#Creating the Confusion matrix
classifier = DecisionTreeClassifier(criterion = 'gini',random_state = 23)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
print(cm)
#accuracy_score (Y_test, Y_pred)

accuracy_dt = accuracy_score(Y_test, Y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy_dt, 2)) + ' %.')

#find the classification
from sklearn.metrics import classification_report 
print(classification_report(Y_test,Y_pred))
print(f1_score(Y_test,Y_pred))

#Visualizing the training set result

#Visualize text representation

from sklearn import tree
text_representation = tree.export_text(classifier)
print(text_representation)

fig = plt.figure(figsize=(25,20))
tree.plot_tree(classifier)

#-----------------------------------------------------------------------------------------------
#2: K-Nearest Neighbour(K-NN) Classifier

# k_nei = KNeighborsClassifier()
# k_nei.fit(X_train,Y_train)

# k_acc = accuracy_score(Y_test,k_nei.predict(X_test))

# print("Train set Accuracy:"+str(accuracy_score(Y_train,k_nei.predict(X_train))*100))
# print("Test Set Accuracy:"+str(accuracy_score(Y_test,k_nei.predict(X_test))*100))

#Creating the Confusion Matrix

classifier = KNeighborsClassifier(n_neighbors=20,p=2,metric='minkowski')
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test,Y_pred)
print(cm)
accuracy_knn = accuracy_score(Y_test, Y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy_knn, 2)) + ' %.')

#find the classification
print(classification_report(Y_test, Y_pred))
print(f1_score(Y_test,Y_pred))

#Cross validation of K-NN model
from sklearn.model_selection import cross_val_score

# creating list of K for KNN
k_list = list(range(1,148))

# creating list of cv scores
cv_scores = []

# perform 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    
# changing to misclassification error
MSE = [1-x for x in cv_scores]

best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)

# Plotting error values against K values

# import seaborn as sns
# # changing to misclassification error
# MSE = [1-x for x in cv_scores]
# plt.figure()
# plt.figure(figsize=(15,10))
# plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
# plt.xlabel('Number of Neighbors K', fontsize=15)
# plt.ylabel('Misclassification Error', fontsize=15)
# sns.set_style("whitegrid")
# plt.plot(k_list, MSE)
# plt.show()



#-----------------------------------------------------------------------------------------------
#3: SVM Classifier

# s_vec = SVC()
# s_vec.fit(X_train,Y_train)

# s_acc = accuracy_score(Y_test,s_vec.predict(X_test))

# print("Train set Accuracy:"+str(accuracy_score(Y_train,s_vec.predict(X_train))*100))
# print("Test Set Accuracy:"+str(accuracy_score(Y_test,s_vec.predict(X_test))*100))

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(X_train, Y_train)

Y_pred = svm.predict(X_test)

print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(X_train, Y_train)))

print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(X_test, Y_test)))

#Creating the confusion matrix

cm = confusion_matrix(Y_test,Y_pred)
print(cm)

accuracy_svm = accuracy_score(Y_test, Y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy_svm, 2)) + ' %.')

#find the classification
print(classification_report(Y_test, Y_pred))
print(f1_score(Y_test,Y_pred))




#Evaluation Table:
models = pd.DataFrame({
    'Algorithm': ['Decision Tree Classifier', 
             'K-NN Classifier',  'SVM Classifier'],
    'f1 score': [0.6075, 0.4642, 0.5901],
    'jaccard Score': [0.4363, 0.3023, 0.3023],
    'Accuracy in %': [accuracy_dt, accuracy_knn, accuracy_svm]
    
})

models.sort_values(by = ['f1 score','jaccard Score','Accuracy in %'], ascending = True)




#Plotting the ROC Curve

# train models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Decision Tree Classifier
model1 = DecisionTreeClassifier()
# knn
model2 = KNeighborsClassifier(n_neighbors=4)
# SVM
model3 = SVC(probability = True)

# fit model
model1.fit(X_train, Y_train)
model2.fit(X_train, Y_train)
model3.fit(X_train, Y_train)

# predict probabilities
pred_prob1 = model1.predict_proba(X_test)
pred_prob2 = model2.predict_proba(X_test)
pred_prob3 = model3.predict_proba(X_test)

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(Y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(Y_test, pred_prob2[:,1], pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(Y_test, pred_prob3[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(Y_test))]
p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=1)

from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(Y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(Y_test, pred_prob2[:,1])
auc_score3 = roc_auc_score(Y_test, pred_prob2[:,1])

print(auc_score1, auc_score2, auc_score3 )

# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Decision Tree Classifier')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
plt.plot(fpr3, tpr3, linestyle='--', color='blue', label='SVM Classifier')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();
