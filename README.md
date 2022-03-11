# Titanic survival prediction
Titanic classification [challenge on Kaggle](https://www.kaggle.com/c/titanic).
Given a dataset of a subset of the Titanic's passengers predict whether they will survive or not.

## Credits
* Maciej Bialoglowski  ([@chemista](https://github.com/chemista))

## Method
Below are provided steps that I followed for this Project.

### 1. **Data visualization**: Data analisys to understand features, missing values, mean values (for further use) and other usefull information.
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
([Jupyters notebook](https://github.com/minio999-kaggle/titanic/blob/dev/playground/eda1.ipynb)) where its done.

We can clearly see that Age has some null values we'll need to fix it. Also we see that Sex is an object so we have to change it to int. Cabin has to much null values so we can bassicly dump it.
- Getting better knowlage about data
([Jupyters notebook](https://github.com/minio999-kaggle/titanic/blob/dev/playground/eda1.ipynb)) where its done.

Here we will want to take mean value of age and give it to null values in it. Also there is no ridiculess thing that we want to get ride of (like age 300)

- Conclusion
We have to process null values in Age. Simplest and used way is to fill them with mean value. We can come to a conclusion that some columns are useless in our prediction's like PassengerId, Name, Cabin, Ticket, Embarked.

### 2. **Preprocessing**: with the knowlage acquired with data visualization, we can apply it to dealing with missing values and specifying features that we want to use in our predictions

- Imputing Age
([Jupyters notebook](https://github.com/minio999-kaggle/titanic/blob/readme/playground/simple_model.ipynb)) where its done.
- Converting Sex 
([Jupyters notebook](https://github.com/minio999-kaggle/titanic/blob/readme/playground/simple_model.ipynb)) where its done.

- Spliting data into Test and Train
([Jupyters notebook](https://github.com/minio999-kaggle/titanic/blob/readme/playground/simple_model.ipynb)) where its done.
- Aplying Preproccesing to data
([Jupyters notebook](https://github.com/minio999-kaggle/titanic/blob/readme/playground/simple_model.ipynb)) where its done.
## Folder Structures
* `\` contains all of setup files
* `\src\app` contains code
* `\playground` contains jupyter notebooks for test purpose

## Installation instructions
1. Install Python and clone this repository
2. Open files, find cloned repository, open terminal inside that folder and use command `./run.sh`

to run the [jupyter](http://jupyter.org/)'s notebooks or mess with it yourself download docker, open powershell and run `.\jupyter-start.ps1`
