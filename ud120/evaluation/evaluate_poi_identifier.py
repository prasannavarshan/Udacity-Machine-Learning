#!/usr/bin/python
# coding=utf-8


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print "accuracy score:", clf.score(features_test, labels_test)

# How many POIs are in the test set for your POI identifier?
# 测试集中有多少 POI？
pred = clf.predict(features_test)

print len([e for e in labels_test if e == 1.0])

# How many people total are in your test set?
print "测试集总人数:", len(labels_test)

# Precision and recall can help illuminate your performance better.
# Use the precision_score and recall_score available in sklearn.metrics to compute those quantities.

# What’s the precision?


print "precision_score", precision_score(labels_test, pred)

# What’s the recall?
print "recall_score", recall_score(labels_test, pred)

# Here are some made-up predictions and true labels for a hypothetical test set; fill in the following boxes to
# practice identifying true positives, false positives, true negatives, and false negatives. Let’s use the convention
#  that “1” signifies a positive result, and “0” a negative.
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print len([e for e in true_labels if e == 0])





# What's the precision of this classifier?
print precision_score(true_labels, predictions)

# What's the recall of this classifier?
print recall_score(true_labels, predictions)
