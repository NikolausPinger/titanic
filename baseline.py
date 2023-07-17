import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import mlflow
from ydata_profiling import ProfileReport
from trail import Trail

import inspect


def dataloader():
    for dirname, _, filenames in os.walk('./input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    train_data = pd.read_csv('./input/train.csv')
    train_data.head()
    test_data = pd.read_csv('./input/test.csv')
    return train_data, test_data


def split(train_data, test_data):
    train, test = train_test_split(train_data, test_size=0.2, random_state=42)
    train.head()
    test.head()

    train.to_csv('./input/train_split.csv')
    test.to_csv('./input/test_split.csv')
    return train



def trainer(train_data):
    #train_data = train_data[:400]
    y = train_data["Survived"]
    features = ["Pclass", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])

    PROFILE_PATH = "./Metadata/Exp1_ML/train_data_report.html"
    profile = ProfileReport(train_data, title="train_data Profiling Report")
    profile.to_file(PROFILE_PATH)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
    with mlflow.start_run():
        with Trail("myProjectAlias") as trail:
            trail.put_hypothesis("Baseline RandomForest training on all samples without the feature `sex`")
            trail.put_artifact(PROFILE_PATH, "profiling_result.html", "data")
            model.fit(X, y)
            mlflow.log_metric("accuracy", model.score(X, y))
            precision = precision_score(y, model.predict(X))
            recall = recall_score(y, model.predict(X))
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_param("random_state", 1)
            get_file()
            trail.put_artifact("./Metadata/Exp1_ML/code.txt", "code", "code")


def precision_score(y, y_pred):
    return np.sum(y == y_pred) / len(y)


def recall_score(y, y_pred):
    return np.sum(y == y_pred) / len(y)


def get_file():
    code_txt = inspect.getsource(inspect.getmodule(inspect.currentframe()))
    with open("./Metadata/Exp1_ML/code.txt", "w") as f:
        f.write(code_txt)


if __name__ == '__main__':
    a, b = dataloader()
    train_data = split(a, b)
    trainer(train_data)
