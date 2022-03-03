'''
main module for app
'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


PATH = './data/train.csv'
df = pd.read_csv(PATH)


def impute_age(dataframe, value):
    '''
    changing NaN values
    '''
    dataframe['Age'] = dataframe["Age"].fillna(value)
    return dataframe


def convert_sex(dataframe):
    '''
    converting sex
    '''
    dataframe['is_male'] = 0
    dataframe.loc[dataframe['Sex'] == 'male', 'is_male'] = 1
    dataframe = dataframe.drop(columns=['Sex'])
    return dataframe


features = ['Age', 'Sex', 'Pclass']
LABEL = 'Survived'

X = df[features]
y = df[LABEL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mean_age = X_train['Age'].mean()

X_train = impute_age(X_train, mean_age)
X_train = convert_sex(X_train)
X_test = impute_age(X_test, mean_age)
X_test = convert_sex(X_test)

clf_model = RandomForestClassifier(max_depth=4, random_state=42)
clf_model.fit(X_train, y_train)
y_predict = clf_model.predict(X_test)
model_acc = accuracy_score(y_test, y_predict)
print(model_acc)
