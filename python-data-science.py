import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# Data input
data = pd.read_csv('startup data.csv')

# Treating missing values
data=data.fillna(method ='ffill')
data=data.fillna(method ='bfill')

#Deleting columns with redundant/duplicate attributes
delcol = {'Unnamed: 0','latitude','longitude','city','Unnamed: 6','name','id','object_id','zip_code','labels','state_code.1','state_code','category_code'}
for col in delcol:
	del data[col]
    
# Binning the attribute 'funding_total_usd' into 6 bins
data['funding_total_usd'] = pd.qcut(data['funding_total_usd'], q=6)

# Extracting the year from date attributes
dataTransform = {'founded_at','closed_at','first_funding_at','last_funding_at'}
for tf in dataTransform:
	data[tf]=pd.DatetimeIndex(data[tf]).year
    
# Rounding float data
rounding = {'age_first_funding_year','age_last_funding_year','age_first_milestone_year','age_last_milestone_year','avg_participants'}
for rd in rounding:
	data[rd]=(data[rd]).apply(np.floor)

# Transforming categorical attribute into binary
dum_money = pd.get_dummies(data.funding_total_usd,prefix = 'funding_total')
data = pd.concat([data,dum_money],axis=1).drop(columns=['funding_total_usd'])


#########################

X = data
y = data.pop('status')

# Splitting the data into a training and a testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.67,random_state=32)
#########################

# Creating a k nearest neighbors classifier
knn = KNeighborsClassifier()

# Setting up GridSearchCV parameters
grid_params1 = {
    'n_neighbors':[3,5,7,9,11],
    'weights' : ['uniform','distance'],
    'metric' : ['euclidean','manhattan']
}
knn_grid = GridSearchCV(knn,grid_params1,cv=5)

# Fitting it to the training set
knn_results = knn_grid.fit(X_train,y_train)

# Print the tuned parameters and score
print("Tuned K Nearest Neighbors Parameters: {}".format(knn_results.best_params_)) 
print("Best score is {}".format(knn_results.best_score_))

# Predicting the cassification for the testing set
y_pred = knn_results.predict(X_test)

# Print the confusion matrix and classification report for our model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Compute predicted probabilities: y_pred_prob1
y_pred_prob1 = knn_results.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr1, tpr1, thresholds = roc_curve(y_test,y_pred_prob1,pos_label = 'closed')

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr1,tpr1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve KNN')
plt.show()

#########################


# Creating a logistic regression classifier
logreg = LogisticRegression(max_iter=10000)

# Setting up GridSearchCV parameters
c_space = np.logspace(-5, 8, 15)
grid_params2 = {'C': c_space}


logreg_grid = GridSearchCV(logreg,grid_params2,cv=5)

# Fitting it to the training set
logreg_results = logreg_grid.fit(X_train,y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_results.best_params_)) 
print("Best score is {}".format(logreg_results.best_score_))

# Predicting the cassification for the testing set
y_pred = logreg_results.predict(X_test)

# Print the confusion matrix and classification report for our model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Compute predicted probabilities: y_pred_prob2
y_pred_prob2 = logreg_results.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr2, tpr2, thresholds
fpr2, tpr2, thresholds = roc_curve(y_test,y_pred_prob2,pos_label = 'closed')

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr2,tpr2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Logistic Regression')
plt.show()

