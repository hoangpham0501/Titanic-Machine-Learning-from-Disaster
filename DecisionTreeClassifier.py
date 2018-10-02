import os
import re
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import tree 


# Setting working enviroment 
path = os.path.expanduser('C:/Relax/Titanic-Machine-Learning-from-Disaster')
os.chdir(path)

# read training and testing data
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

def preprocessingData(df):
	# averAge = int(df.Age.median())
	# fare = float(df_test.Fare.median())

	# Fill the empty fields
	# df['Age'] = df['Age'].fillna(averAge)
	# df['Fare'] = df['Fare'].fillna(fare)

	# Replace string fields by number
	df['Sex'] = df['Sex'].replace('male', 1)
	df['Sex'] = df['Sex'].replace('female', 0)


	# Add three more features "FamilySize" and "Master"
	# FamilySize = SibSp + Parch.
	df['FamilySize'] = df['SibSp'] + df['Parch']

	# Minor = 1 if 'Master' title appears in name and Minor = 0 if 'Master' does not appear in name. 
	df['Minor'] = 0
	for i in range(len(df.Name)):
		if "Master" in df.Name[i]:
			df.Minor[i] = 1

	df['FamilyOneSurvived'] = 0
	df['FamilyAllDied'] = 0

	df['Surname'] =  df.Name.str.extract("([A-Z]\w{0,})")

	for i in range(len(df.Surname)):
		for j in range(i+1, len(df.Surname)):
			if df.Surname[i] == df.Surname[j] and (df.Survived[i] == 1 or df.Survived[j]):
				df.FamilyOneSurvived[i] = 1
				df.FamilyOneSurvived[j] = 1
	
	for i in range(len(df.Surname)):
		for j in range(i+1, len(df.Surname)):
			if df.Surname[i] == df.Surname[j] and df.FamilyOneSurvived[i] == 0:
				df.FamilyAllDied[i] = 1
				df.FamilyAllDied[j] = 1

	# Drop unnecessary features
	df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked'], axis =1)
	# print(df)
	return df

def preprocessingTestData(df):
	averAge = int(df.Age.median())
	fare = float(df_test.Fare.median())

	# Fill the empty fields
	df['Embarked'] = df['Embarked'].fillna("S")
	df['Age'] = df['Age'].fillna(averAge)
	df['Fare'] = df['Fare'].fillna(fare)

	# Replace string fields by number
	df['Sex'] = df['Sex'].replace('male', 1)
	df['Sex'] = df['Sex'].replace('female', 0)

	# Add three more features "FamilySize" and "Master"
	# FamilySize = SibSp + Parch.
	df['FamilySize'] = df['SibSp'] + df['Parch']

	# Minor = 1 if 'Master' title appears in name and Minor = 0 if 'Master' does not appear in name. 
	df['Minor'] = 0
	for i in range(len(df.Name)):
		if "Master" in df.Name[i]:
			df.Minor[i] = 1
	df['Surname'] =  df.Name.str.extract("([A-Z]\w{0,})")
	# Drop unnecessary features
	df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked'], axis =1)
	return df

def addTwoFeatures(df1, df2):
	df2['FamilyOneSurvived'] = 0
	df2['FamilyAllDied'] = 0
	for i in range(len(df2.Surname)):
		for j in range(len(df1.Surname)):
			if df2.Surname[i] == df1.Surname[j]:
				if df1.FamilyOneSurvived[j] == 1:
					df2.FamilyOneSurvived[i] = 1
				
				if df1.FamilyAllDied[j] == 1:
					df2.FamilyAllDied[i] = 1 
	return df2

if __name__ == '__main__':
	# Get Passenger ID in test data 
	# It will be used to add into final result
	id = df_test['PassengerId']
	id_df = pd.DataFrame(id)
	
	# Preprocessing data	
	train_df = preprocessingData(df)
	test_df = preprocessingTestData(df_test)
	test_df = addTwoFeatures(train_df, test_df)

	train_df = train_df.drop(['Surname'], axis = 1)
	test_df = test_df.drop(['Surname'], axis = 1)
	train_df.to_csv("data.csv")
	test_df.to_csv("testdata.csv")

	# Init train, test and validate data from training data set
	train, validate = np.split(train_df.sample(frac=1), [int(.8*len(train_df))])

	y_train = train['Survived']
	x_train = train.drop(['Survived'], axis = 1)

	y_validate = validate['Survived']
	x_validate = validate.drop(['Survived'], axis = 1)

	tree_model = tree.DecisionTreeClassifier(max_depth=3)
	tree_model.fit(x_train, y_train)
	tree_acc = tree_model.score(x_validate, y_validate) * 100

	accuracy = tree_acc
	print("The best validation accuracy= ", accuracy)
	predicted = pd.DataFrame({'Survived': tree_model.predict(test_df)})

	# Join predicted into result dataframe and write result as a CSV file
	result = id_df.join(predicted)
	result.to_csv("result_final.csv", index = False)


# Average validation accuracy = 83.24022346368714
# Kaggle score = 0.79904