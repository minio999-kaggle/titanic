'''
main module for app
'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


RANDOM_STATE = 42
PATH = './data/train.csv'
df_raw = pd.read_csv(PATH)


def impute_age(df, value):
    '''
    Replaces Nulls in column "Age" of a dataframe with the passed value

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
        value (float): Value used for imputation
    Returns:
        pandas.DataFrame
    '''

    df['Age'] = df["Age"].fillna(value)
    return df


def convert_sex(df):
    '''
    Replacing sex in column "Sex" of a dataframe to 1 if it's male and 0 if it's female

    Parameters:
        dataframe (pandas.DataFrame): Dataframe on which to operate
    Returns:
        pandas.DataFrame
    '''

    df['is_male'] = 0
    df.loc[df['Sex'] == 'male', 'is_male'] = 1
    df = df.drop(columns=['Sex'])
    return df



features = ['Age', 'Sex', 'Pclass']
LABEL = 'Survived'

X = df_raw[features]
y = df_raw[LABEL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

mean_age = X_train['Age'].mean()

X_train = impute_age(X_train, mean_age)
X_train = convert_sex(X_train)
X_test = impute_age(X_test, mean_age)
X_test = convert_sex(X_test)

clf_model = RandomForestClassifier(max_depth=4, random_state=RANDOM_STATE)
clf_model.fit(X_train, y_train)
y_predict = clf_model.predict(X_test)
model_acc = accuracy_score(y_test, y_predict)
print(model_acc)
