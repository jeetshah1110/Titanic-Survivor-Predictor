import  pandas as pd
import numpy as np
import  seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#plot graph for survived and non survived passengers
titanic_data=pd.read_csv("titanic.csv")
figure()
target="Survived"
countplot(data=titanic_data,x=target).set_title("survived and non survived passengers")
show()

#plot graph for survived and non survived passengers according to their gender
figure()
target="Survived"
countplot(data=titanic_data,x=target,hue='Sex').set_title("survived and non survived passengers based on gender")
show()


#plot graph for survived and non survived passengers according to their Class
figure()
countplot(data=titanic_data,x=target,hue='Pclass').set_title("survived and non survived  passengers based on class")
show()

#Encode Sex Field and remove  one field
print("value for Sex after removing one field")
Sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
print(Sex.head())

#Encode Pclass Field and remove  one field
print("value for Plcass after removing one field")
Pclass=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
print(Pclass.head())

#Remove old columns and add new columns
print("values of dataset after removing old columns and adding new columns")
titanic_data=pd.concat([titanic_data,Sex,Pclass],axis=1)
titanic_data.drop(["Sex","SibSp","Parch","Embarked","Name","Ticket","Cabin"],axis=1,inplace=True)
titanic_data=titanic_data[pd.notnull(titanic_data["Age"])]
print(titanic_data.head())


#create input data and targets
x=titanic_data.drop("Survived",axis=1)
y=titanic_data["Survived"]

#divide dataset into training and testing
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5)

#create logistic regression model
logmodel=LogisticRegression()
logmodel.fit(xtrain,ytrain)
prediction=logmodel.predict(xtest)


print("classification report using logistic regression is:")
print(classification_report(ytest,prediction))

print("confusion matrix using logistic regression is:")
print(confusion_matrix(ytest,prediction))

print("accuracy using logistic regression is:")
print(accuracy_score(ytest,prediction))
