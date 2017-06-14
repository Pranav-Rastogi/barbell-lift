import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split


'''
Convert the csv file to a pandas DataFrame. The read_csv function might give
a DtypeWarning due to the large size of dataset. Passing low_memory=False as
an argument will silence the warning.
LabelEncoder is used to map classe from object to int.
'''
df = pd.read_csv("pml-training.csv", low_memory=False)
le = preprocessing.LabelEncoder()

rows, cols = df.shape
print("The DataFrame consists of {} rows and {} columns\n\n".format(rows, cols))

'''
Drop the columns that have null values in them. there are 100 of them.
Then dropsome extra columns that do not provie with any valuable data.
fit_transform() does the actual mapping of classe from object to int.
'''
df.dropna(axis=1, inplace=True)
df.drop(['Unnamed: 0', 'raw_timestamp_part_1', 'cvtd_timestamp', 'user_name',
	'raw_timestamp_part_2', 'new_window','num_window'], axis=1, inplace=True)
df['classe'] = le.fit_transform(df['classe'])

rows, cols = df.shape
print("The DataFrame now consists of {} rows and {} columns\n\n".format(rows, cols))

'''
x now consists of all the feature columns and y consists of the target/label
column.
'''
x = df.drop('classe', axis=1)
y = df.classe
x_train, __, y_train, __ = train_test_split(x, y, test_size=0.2)

#Already explained in Readme.md
clf = RandomForestClassifier(min_impurity_split=0.1, n_jobs = -1, max_depth=15)
score = cross_val_score(clf, x, y, cv=10)
print("Mean k-fold cross_validation score for k = 10:", score.mean())
score = cross_val_score(clf, x, y, cv=5)
print("Mean k-fold cross_validation score for k = 5:", score.mean())

input("\n\nPress enter to exit.")
