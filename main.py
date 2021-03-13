import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

# we have two sets (train and test) but we'll concat them to make things faster and easier.
train_set = pd.read_csv('../input/video-games-rating-by-esrb/Video_games_esrb_rating.csv')
test_set = pd.read_csv('../input/video-games-rating-by-esrb/test_esrb.csv')

data = pd.concat([train_set,test_set],axis=0)
data.head()


# first let's check is there any null data here
data.isnull().sum()

data.info()

x = data.iloc[:,1:-1]
y = data["esrb_rating"]

x.head()

y.head()

for feature in x.columns:
    print(f"{feature}=>   {x[feature].unique()}")
    
# all features are binary so we won't do something new to x, let's check y.
data2int = {category:i for i,category in enumerate(y.unique())}
int2data = {i:category for i,category in enumerate(y.unique())}

data2int
x = np.asarray(x)
y = np.asarray([data2int[val] for val in y.values])

print(x.shape)
print(y[:5])

# and let's split train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# now we'll define a function to choose the best algorithm
def evaluateAlgorithm(cls,name,is_created=False):
    print("Now evaluating {} algorithm".format(name))
    start = time.time()
    if not is_created:
        clf = cls()
    else:
        clf = cls
    clf.fit(x_train,y_train)
    y_true = y_test
    y_pred = clf.predict(x_test)
    
    end = time.time()
    process = round(end - start,2)
    print("Model {} fitted".format(name))
    
    accuracy = round(accuracy_score(y_true=y_true,y_pred=y_pred)
                    ,2) * 100
    conf_matrix = confusion_matrix(y_true=y_true,y_pred=y_pred)
    
    plt.subplots(figsize=(5,5))
    sns.heatmap(conf_matrix,linewidths=1.5,annot=True,fmt=".1f")
    plt.xlabel("Predicted  Label")
    plt.ylabel("True Label")
    plt.show()
    
    print(f"Accuracy of {name} is %{accuracy}, evaluating finished. Model returned")
    
    return clf

svm_clf = evaluateAlgorithm(SVC,"SVM")

rfc = evaluateAlgorithm(RandomForestClassifier,"RFC")

adaboost = evaluateAlgorithm(AdaBoostClassifier(base_estimator=RandomForestClassifier()),
                             "AdaBoost",
                             True
                            )

# best model is Random Forest Classifier with %88 accuracy. Let's pickle it.
import pickle
pickle.dump(rfc,open("randomforest.pkl",mode="wb"))
print("Pickled truly")

import json
json.dump(int2data,open("int2data.json",mode="w"))
json.dump(data2int,open("data2int.json",mode="w"))
