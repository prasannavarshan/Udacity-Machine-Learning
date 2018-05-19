#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import numpy as np

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print data_dict

# temp = []
# for key, value in data_dict.items():
#     if value.get('salary') == 26704229:
#         print key

# print value.get('salary')

# temp.append((key, value.get('salary')))
# name, salary = zip(*temp)
# name = np.reshape(np.array(name), (len(name), 1))
# salary = np.reshape(np.array(salary), (len(salary), 1))
# salary.sort()
# print salary

features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)

for key, value in data_dict.items():
    if (value.get('salary') > 1000000) and (value.get('bonus') > 5000000):
        print key, value.get('salary'), value.get('bonus')

data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
