# Titanic survival prediction
Titanic classification [challenge on Kaggle](https://www.kaggle.com/c/titanic).
Given a dataset of a subset of the Titanic's passengers predict whether they will survive or not.

## Credits
* Maciej Bialoglowski  ([@chemista](https://github.com/chemista))

## Method
Below are provided steps that I followed for this Project.

### 1. **Data visualization**: Data analisys to understand features,missing values, mean values (for further use) and other usefull information.
- Understanding features
    - **PassengerId**: ID of passenger
    - **Survived**: Value specifying if passanger survived ("1") or not ("0")
    - **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
    - **Name**: Full name of passanger
    - **Age**: Age of passenger
    - **SibSp**: Amount of sibling/spouses aboard the Titanic
    - **Parch**: Amount of parent/children aboard the Titanic
    - **Ticket**: Ticket Number
    - **Fare**: Passenger fare
    - **Cabin**: Cabin number
    - **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- Looking for null values
```
path = "../data/train.csv"
df = pd.read_csv(path)
df.info()
```
| # |  Column   |    Non-Null Count | Dtype |  
| :-: | :-: | :-: | :-: |
| 0 |  PassengerId | 891 non-null  |  int64 | 
| 1 |  Survived  |   891 non-null  |  int64 | 
| 2 |  Pclass    |   891 non-null  |  int64 | 
| 3 |  Name      |   891 non-null  |  object | 
| 4 |  Sex       |   891 non-null  |  object |
| 5 |  Age       |   714 non-null  |  float64 |
| 6 |  SibSp     |   891 non-null  |  int64 | 
| 7 |  Parch     |   891 non-null  |  int64 | 
| 8 | Ticket     |   891 non-null  |  object |
| 9 | Fare       |   891 non-null  |  float64 |
| 10 | Cabin      |   204 non-null  |  object |
| 11 | Embarked   |   889 non-null  |  object |

We can clearly see that Age has some null values we'll need to fix it. Also we see that Sex is an object so we have to change it to int. Cabin has to much null values so we can bassicly dump it.
- Getting better knowlage about data
```
df.describe()
```
| # | PassengerId |	Survived | Pclass | Age | SibSp | Parch | Fare | 
| :-:  | :-:  | :-:  | :-: | :-: | :-: | :-: | :-: | 
| count | 891.000000 | 891.000000 | 891.000000 | 714.000000 | 891.000000 | 891.000000 | 891.000000 | 
| mean | 446.000000 | 0.383838 | 2.308642 |	29.699118 |	0.523008 | 0.381594 | 32.204208 | 
| std |	257.353842 | 0.486592 |	0.836071 |	14.526497 |	1.102743 |	0.806057 |	49.693429 | 
| min |	1.000000 |	0.000000 |	1.000000 |	0.420000 |	0.000000 |	0.000000 |	0.000000 | 
| 25% |	223.500000 | 0.000000 |	2.000000 |	20.125000 |	0.000000 |	0.000000 |	7.910400 | 
| 50% |	446.000000 | 0.000000 |	3.000000 |	28.000000 |	0.000000 |	0.000000 |	14.454200 | 
| 75% |	668.500000 | 1.000000 |	3.000000 |	38.000000 |	1.000000 |	0.000000 |	31.000000 | 
| max |	891.000000 | 1.000000 |	3.000000 |	80.000000 |	8.000000 |	6.000000 |	512.329200 | 

Here we will want to take mean value of age and give null values it. Also there is no ridiculess thing that we want to get ride of (like age 300)

- Conclusion
We have to process null values in Age simplest and used way is to fill them with mean value. We can come to a conclusion that some columns are useless in our prediction's like PassengerId, Name, Cabin, Ticket, Embarked.

### 2. **Preprocessing**: with the knowlage acquired with data visualization, we can apply it to dealing with missing values and specifying features that we want to use in our predictions

- Imputing Age
```
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
```

- Converting Sex 
```
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
```

- Spliting data into Test and Train
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
```

- Aplying Preproccesing to data
```
def transform_data(df, mean_age_value):
    '''
    Applying data cleaning functions to data sets

    Paramters:
        dataframe (pandas.DataFrame): Dataframe on which to operate
        mean_age (float): Mean age of training data set
    Retruns:
        pandas.DataFrame
    '''

    df = impute_age(df, mean_age_value)
    df = convert_sex(df)
    return df

X_train = transform_data(X_train, mean_age)
X_test = transform_data(X_test, mean_age)
```

## Folder Structures
* `\` contains all of setup files
* `\src\app` contains code
* `\playground` contains jupyter notebooks for test purpose

## Installation instructions
1. Install Python and clone this repository
2. Open files, find cloned repository, open terminal inside that folder and use comman `./run.sh`

to run the [jupyter](http://jupyter.org/)'s notebooks or mess with it yourself download docker, open powershell and run `.\jupyter-start.ps1`
