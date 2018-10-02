import os
import re
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier 


# Setting working enviroment 
path = os.path.expanduser('C:/Users/pvhoang/Desktop/Titanic_dataset')
os.chdir(path)

# read training and testing data
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

def preprocessingData(df):
	averAge = int(df.Age.median())
	fare = float(df_test.Fare.median())

	# Fill the empty fields
	df['Age'] = df['Age'].fillna(averAge)
	df['Fare'] = df['Fare'].fillna(fare)

	# Replace string fields by number
	df['Sex'] = df['Sex'].replace('male', 1)
	df['Sex'] = df['Sex'].replace('female', 0)

	# Add two more features "FamilySize" and "Minor"
	# FamilySize = SibSp + Parch.
	df['FamilySize'] = df['SibSp'] + df['Parch']

	# Minor = 1 if 'Master' title appears in name and Minor = 0 if 'Master' does not appear in name. 
	df['Minor'] = 0
	for i in range(len(df.Name)):
		if "Master" in df.Name[i]:
			df.Minor[i] = 1

	# Drop unnecessary features
	df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked'], axis =1)
	print(df)
	return df


if __name__ == '__main__':
	# Get Passenger ID in test data 
	# It will be used to add into final result
	id = df_test['PassengerId']
	id_df = pd.DataFrame(id)
	
	# Preprocessing data	
	train_df = preprocessingData(df)
	test_df = preprocessingData(df_test)

	# Init train, test and validate data from training data set
	train, validate = np.split(train_df.sample(frac=1), [int(.8*len(train_df))])

	y_train = train['Survived']
	x_train = train.drop(['Survived'], axis = 1)

	y_validate = validate['Survived']
	x_validate = validate.drop(['Survived'], axis = 1)

	tree_model = tree.DecisionTreeClassifier(max_depth=3)
	tree_model.fit(x_train, y_train)
	tree_acc = tree_model.score(x_validate, y_validate) * 100
			
	# accuracy = tree_acc
	predicted = pd.DataFrame({'Survived': tree_model.predict(test_df)})
			
	accuracy = tree_acc
	print("The best validation accuracy= ", accuracy)

	# Join predicted into result dataframe and write result as a CSV file
	result = id_df.join(predicted)
	result.to_csv("result_final.csv", index = False)


# Average validation accuracy =  82%
# Score on kaggle = 79%
