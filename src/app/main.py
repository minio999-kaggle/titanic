'''
main module for app
'''


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
MIN_SAMPLE_SPLIT=4
MIN_SAMPLES_LEAF=5
N_ESTIMATORS=100
N_SPLITS = 5
USELESS_FEATURES = ["PassengerId", "Name", "Ticket", "Cabin",
                    "Parch", "SibSp", 'Embarked', 'Fare']
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

def count_relatives_on_board(df):
    '''
    Counting Relatives on board based of sibsp and parch columns

    Paramters:
        dataframe (pandas.DataFrame): Dataframe on which to operate
    Retruns:
        pandas.DataFrame
    '''

    df["RelativesOnboard"] = df["SibSp"] + df["Parch"]
    return df

def set_title(df):
    '''
    Changing name titles to cryptonims

    Paramters:
        dataframe (pandas.DataFrame): Dataframe on which to operate
    Retruns:
        pandas.DataFrame
    '''

    mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
      'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
    df.replace({'Title': mapping}, inplace=True)
    return df

def transform_data(df, mean_age_value):
    '''
    Applying data cleaning functions to data sets

    Paramters:
        dataframe (pandas.DataFrame): Dataframe on which to operate
        mean_age (float): Mean age of training data set
    Retruns:
        pandas.DataFrame
    '''
    df = set_title(df)
    df = count_relatives_on_board(df)
    df = impute_age(df, mean_age_value)
    df = convert_sex(df)
    return df

def main():
    '''
    Main Function
    '''

    LABEL = 'Survived'

    X = df_raw.copy()
    X = X.drop('Survived', axis=1)
    y = df_raw['Survived']

    k_fold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = []

    for train_index, test_index in k_fold.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        mean_age = X_train['Age'].mean()

        X_train = transform_data(X_train, mean_age)

        X_test = transform_data(X_test, mean_age)

        X_train.drop(USELESS_FEATURES, inplace=True,axis=1)
        X_test.drop(USELESS_FEATURES, inplace=True,axis=1)
        #print(X_train.columns)
        label_encoder = LabelEncoder()
        X_train['Title'] = label_encoder.fit_transform(X_train['Title'])
        X_test['Title'] = label_encoder.fit_transform(X_test['Title'])

        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, bootstrap=True, criterion='entropy',
                                    min_samples_leaf=MIN_SAMPLES_LEAF,
                                    min_samples_split=MIN_SAMPLE_SPLIT, random_state=RANDOM_STATE)

        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)

        acc_score = round(accuracy_score(y_test, y_predict),3)

        print(acc_score)

        scores.append(acc_score)

    print()
    print("Average:", round(100*np.mean(scores), 1), "%")
    print("Std:", round(100*np.std(scores), 1), "%")

if __name__ == "__main__":
    main()
