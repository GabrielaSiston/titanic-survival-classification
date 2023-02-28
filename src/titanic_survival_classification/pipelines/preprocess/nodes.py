"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.4
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def dataSplit(df):

    y = np.asarray(list(df['Survived']))
    X = df.drop('Survived', axis=1)

    return X, y

def preprocess_data(train, test):

    age_median = train['Age'].median()
    train['Age'] = train['Age'].fillna(age_median)
    train = train.drop(columns=['Cabin','Ticket', 'PassengerId', 'Name'])
    train_num =  pd.DataFrame(columns=['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    train_num = pd.get_dummies(data = train,
        columns = ['Sex', 'Embarked'],
        prefix = ['Sex', 'Embarked']
    )

    #transformações dataset test
    age_median = test['Age'].median()
    test['Age'] = test['Age'].fillna(age_median)
    test = test.drop(columns=['Cabin','Ticket', 'PassengerId', 'Name'])
    test = test.dropna(subset=['Fare']).reset_index(drop=True)
    test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    test_num =  pd.DataFrame(columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    test_num = pd.get_dummies(data = test,
        columns = ['Sex', 'Embarked'],
        prefix = ['Sex', 'Embarked']
    )
    return train_num, test_num

def predict(train_num, test_num):
    
    X, y = dataSplit(train_num)

    mean_acc=0
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    clf = RandomForestClassifier(max_features= "sqrt",  random_state=3232)
    bestModel = None
    bestAcc = 0
    i = 0
    for train_index, val_index in kf.split(X):
        i += 1
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_val)
        acc = (100*accuracy_score(y_val, predictions))
        print("Fold: %i" % i  + " - Accuracy: %s%%" % acc)
        if acc > bestAcc:
            bestModel = clf
            bestAcc = acc
        mean_acc += acc
    #print(classification_report(y_val, predictions))
    print("Acuracia Média: %s%%" % (mean_acc/i))
    print("Melhor Acuracia %s%%" % bestAcc)
    
    predictions_test = bestModel.predict(test_num)
    test_num['predictedValues'] = predictions_test
    result = test_num#['predictedValues'] 

    return result


