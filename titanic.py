import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

train = pd.read_csv('train.csv')
train.columns
train.head()
train.describe()

# 欠損値の確認
train.isnull().sum()

# --------------------------------------------
# まずはRandomForestでやてみる．
# Cabinは欠損が多すぎるので捨てる
# Ageは中央値でも入れておくか
def make_train_RandomForest(train):
    train_filled_median = train.copy()
    train_filled_median = train_filled_median.fillna(value={'Age': train.describe()['Age']['50%'], 'Embarked': 'S'}, inplace=True)
    train_filled_median = train_filled_median.drop(['Cabin', 'Embarked', 'Name', 'Ticket'], axis=1)
    train_filled_median = train_filled_median.replace({'male':0, 'female':1})
    return train_filled_median

train_RandomForest = make_train_RandomForest(train)
# train_RandomForest.isnull().sum()

X = train_RandomForest.drop(['PassengerId', 'Survived'], axis=1)
y = train_RandomForest['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier().fit(X_train, y_train)
rfc.score(X_test, y_test)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, max_depth=3).fit(X_train, y_train)
gbc.score(X_test, y_test)


test = pd.read_csv('test.csv')
test = make_train_RandomForest(test)
test = test.fillna(value={"Fare": test.describe().Fare["50%"]})
# test.isnull().sum()


rfc_res = pd.DataFrame(rfc.predict(test.drop(['PassengerId'], axis=1)), columns=['Survived'])
result = pd.concat([test.PassengerId, rfc_res], axis=1)
result.to_csv("rfc.csv", index=False)


gbc_res = pd.DataFrame(gbc.predict(test.drop(['PassengerId'], axis=1)), columns=["Survived"])
result = pd.concat([test.PassengerId, gbc_res], axis=1)
result.to_csv("gbc.csv", index=False)
