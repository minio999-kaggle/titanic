'''
main module
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'
N_SPLITS = 5
RANDOM_STATE = 42


def extracting_title_age_imputing(data_df, train_df, test_df):
    '''
    Function to extract title and impute age

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
        dataframe (pandas.DataFrame): DataFrame on which to train
        dataframe (pandas.DataFrame): DataFrame on which to test
    '''
    data_df['Title'] = data_df['Name']

    for name_string in data_df['Name']:
        data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

    mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col':
                'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
                'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt':
                'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
    data_df.replace({'Title': mapping}, inplace=True)

    titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
    for title in titles:
        age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
        data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

    train_df['Age'] = data_df['Age'][:891]
    test_df['Age'] = data_df['Age'][891:]
    data_df.drop('Title', axis = 1, inplace = True)

def family_size(data_df, train_df, test_df):
    '''
    Function to combine Parch and Sibsp columns to get family size on board

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
        dataframe (pandas.DataFrame): DataFrame on which to train
        dataframe (pandas.DataFrame): DataFrame on which to test
    '''
    data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']
    train_df['Family_Size'] = data_df['Family_Size'][:891]
    test_df['Family_Size'] = data_df['Family_Size'][891:]

def spliting_name(data_df):
    '''
    Function to split name column to name and last name

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
    '''
    data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
    
def imputing_fare(data_df):
    '''
    Function to impute fare column

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
    '''
    data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)


def family_survival(data_df, train_df, test_df):
    '''
    Function to get family survival rate

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
        dataframe (pandas.DataFrame): DataFrame on which to train
        dataframe (pandas.DataFrame): DataFrame on which to test
    '''
    DEFAULT_SURVIVAL_VALUE = 0.5
    data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

    for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                            'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

        if (len(grp_df) != 1):
            # A Family group is found.
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
    for _, grp_df in data_df.groupby('Ticket'):
        if (len(grp_df) != 1):
            for ind, row in grp_df.iterrows():
                if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                    smax = grp_df.drop(ind)['Survived'].max()
                    smin = grp_df.drop(ind)['Survived'].min()
                    passID = row['PassengerId']
                    if (smax == 1.0):
                        data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                    elif (smin==0.0):
                        data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

    # # Family_Survival in TRAIN_DF and TEST_DF:
    train_df['Family_Survival'] = data_df['Family_Survival'][:891]
    test_df['Family_Survival'] = data_df['Family_Survival'][891:]

def fare_bin(data_df, train_df, test_df):
    '''
    Function to get fare bin and encode it

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
        dataframe (pandas.DataFrame): DataFrame on which to train
        dataframe (pandas.DataFrame): DataFrame on which to test
    '''
    data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)
    data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)

    label = LabelEncoder()
    data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])

    train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]
    test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]

    train_df.drop(['Fare'], axis=1, inplace=True)
    test_df.drop(['Fare'], axis=1, inplace=True)

def age_bin(data_df, train_df, test_df):
    '''
    Function to get age bin and encode it

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
        dataframe (pandas.DataFrame): DataFrame on which to train
        dataframe (pandas.DataFrame): DataFrame on which to test
    '''
    data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)

    label = LabelEncoder()
    data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])

    train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]
    test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]

    train_df.drop(['Age'], axis=1, inplace=True)
    test_df.drop(['Age'], axis=1, inplace=True)

def replacing_sex(train_df, test_df):
    '''    
    Function to replace sex to male or female

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to train
        dataframe (pandas.DataFrame): DataFrame on which to test
    '''
    train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
    test_df['Sex'].replace(['male','female'],[0,1],inplace=True)

def aplying_preprocesing(data_df,train_df, test_df):
    '''
    Function to apply preprocesing and drop columns

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
        dataframe (pandas.DataFrame): DataFrame on which to train
        dataframe (pandas.DataFrame): DataFrame on which to test
    '''
    extracting_title_age_imputing(data_df, train_df, test_df)
    family_size(data_df, train_df, test_df)
    spliting_name(data_df)
    imputing_fare(data_df)
    family_survival(data_df, train_df, test_df)
    fare_bin(data_df, train_df, test_df)
    age_bin(data_df, train_df, test_df)
    replacing_sex(train_df, test_df)
    train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
               'Embarked'], axis = 1, inplace = True)
    test_df.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
              'Embarked'], axis = 1, inplace = True)

def main():
    '''
    Main function
    '''
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    data_df = pd.concat([train_df, test_df])

    aplying_preprocesing(data_df, train_df, test_df)

    X = train_df.drop('Survived', axis=1)
    y = train_df['Survived']
    X_test_kaggle = test_df.copy()

    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)
    X_test = std_scaler.transform(X_test)

    k_fold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = []

    for train_index, test_index in k_fold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2,
                           weights='uniform')
        knn.fit(X_train, y_train)
        y_predict = knn.predict(X_test)

        acc_score = round(accuracy_score(y_test, y_predict),3)

        print(acc_score)

        scores.append(acc_score)

    print()
    print("Average:", round(100*np.mean(scores), 1), "%")
    print("Std:", round(100*np.std(scores), 1), "%")


if __name__ == '__main__':
    main()
