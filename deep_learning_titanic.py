#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import mlflow
from trail import Trail
from ydata_profiling import ProfileReport
import inspect

tf.disable_v2_behavior()

def get_file():
    code_txt = inspect.getsource(inspect.getmodule(inspect.currentframe()))
    with open("./Metadata/Exp2_DL/code.txt", "w") as f:
        f.write(code_txt)

def preprocess_data():
    train_data = pd.read_csv(r"./input/train.csv")
    test_data = pd.read_csv(r"./input/test.csv")
    # Feature Engineering

    nan_columns = ["Age", "SibSp", "Parch"]

    train_data = nan_padding(train_data, nan_columns)
    test_data = nan_padding(test_data, nan_columns) 

    #save PassengerId for evaluation
    test_passenger_id=test_data["PassengerId"]
    not_concerned_columns = ["PassengerId", "Name", "Ticket", "Fare", "Cabin", "Embarked", "Sex"]
    train_data = drop_not_concerned(train_data, not_concerned_columns)
    test_data = drop_not_concerned(test_data, not_concerned_columns)

    dummy_columns = ["Pclass"]
    train_data = dummy_data(train_data, dummy_columns)
    test_data = dummy_data(test_data, dummy_columns)

    #train_data = sex_to_int(train_data)
    #test_data = sex_to_int(test_data)
    train_data.head()

    train_data = normalize_age(train_data)
    test_data = normalize_age(test_data)

    print(train_data.head())

    train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)
    print("train_x:{}".format(train_x.shape))
    print("train_x_type:{}".format(type(train_x)))

    print("train_y:{}".format(train_y.shape))
    print("train_y content:{}".format(train_y[:3]))

    print("valid_x:{}".format(valid_x.shape))
    print("valid_y:{}".format(valid_y.shape))

    PROFILE_PATH = "./Metadata/Exp2_DL/train_data_report.json"
    profile = ProfileReport(train_data, title="train_data Profiling Report")
    profile.to_file(PROFILE_PATH)

    return train_x, train_y, valid_x, valid_y, test_data, test_passenger_id

def nan_padding(data, columns):
    for column in columns:
        imputer=SimpleImputer()
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data


def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)


def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


def sex_to_int(data):
    le = LabelEncoder()
    le.fit(["male","female"])
    data["Sex"]=le.transform(data["Sex"]) 
    return data

def normalize_age(data):
    scaler = MinMaxScaler()
    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
    return data

def split_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y


# Build Neural Network
from collections import namedtuple

def build_neural_network(train_x, train_y, hidden_units=10):
    tf.compat.v1.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training=tf.Variable(True,dtype=tf.bool)
    
    initializer = tf.keras.initializers.glorot_normal
    fc = tf.layers.dense(inputs, hidden_units, activation=None,kernel_initializer=initializer)
    fc=tf.layers.batch_normalization(fc, training=is_training)
    fc=tf.nn.relu(fc)
    
    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(cross_entropy)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'learning_rate','is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph

def get_batch(data_x,data_y,batch_size=32):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        
        yield batch_x,batch_y

def train(train_x, train_y):
    model = build_neural_network(train_x, train_y)
    epochs = 800
    train_collect = 50
    train_print=train_collect*2

    learning_rate_value = 0.001
    batch_size=16

    x_collect = []
    train_loss_collect = []
    train_acc_collect = []
    valid_loss_collect = []
    valid_acc_collect = []

    saver = tf.train.Saver()
    PROFILE_PATH = "./Metadata/Exp2_DL/train_data_report.json"

    with tf.Session() as sess:
        with mlflow.start_run():
            with Trail("myProjectAlias") as run:
                run.put_hypothesis("deep-learning 800 epochs without Feature 'Sex'")
                run.put_artifact(PROFILE_PATH, "profiling_result.html", "data")

                mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate_value})
                sess.run(tf.global_variables_initializer())
                iteration=0
                for e in range(epochs):
                    for batch_x,batch_y in get_batch(train_x,train_y,batch_size):
                        iteration+=1
                        feed = {model.inputs: train_x,
                                model.labels: train_y,
                                model.learning_rate: learning_rate_value,
                                model.is_training:True
                            }

                        train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)
                        
                        if iteration % train_collect == 0:
                            x_collect.append(e)
                            train_loss_collect.append(train_loss)
                            train_acc_collect.append(train_acc)

                            if iteration % train_print==0:
                                print("Epoch: {}/{}".format(e + 1, epochs),
                                "Train Loss: {:.4f}".format(train_loss),
                                "Train Acc: {:.4f}".format(train_acc))
                                mlflow.log_metric("train_loss", train_loss)
                                mlflow.log_metric("train_acc", train_acc)  
                            feed = {model.inputs: valid_x,
                                    model.labels: valid_y,
                                    model.is_training:False
                                }
                            val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                            valid_loss_collect.append(val_loss)
                            valid_acc_collect.append(val_acc)
                            
                            if iteration % train_print==0:
                                print("Epoch: {}/{}".format(e + 1, epochs),
                                "Validation Loss: {:.4f}".format(val_loss),
                                "Validation Acc: {:.4f}".format(val_acc))
                                mlflow.log_metric("loss", val_loss)
                                mlflow.log_metric("accuracy", val_acc)
                saver.save(sess, "./titanic.ckpt")
                get_file()
                run.put_artifact("./Metadata/Exp2_DL/code.txt", "code", "code")



if __name__ == "__main__":
    train_x, train_y, valid_x, valid_y, test_data, test_passenger_id = preprocess_data()
    train(train_x, train_y)
