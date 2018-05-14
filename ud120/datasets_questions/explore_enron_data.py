#!/usr/bin/python
# coding=utf-8

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy as np

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
poi_names_txt = "../final_project/poi_names.txt"
# print "样本个数:", len(enron_data)
# poi = []
# for value in enron_data.values():
#     if value.get('poi'):
#         poi.append(1)
#
# print "poi为True的个数:", len(poi)
#
# print "特征个数:", len(enron_data.get('METTS MARK'))


# my_poi = []
# for line in open(poi_names_txt):
#     if line.startswith('('):
#         my_poi.append(1)
# print "自己整理的数据集中poi的个数:", len(my_poi)


# for kv in enron_data.items():
#     print kv


# for key, values in enron_data.items():
#     if key == ('PRENTICE JAMES'.upper()):
#         print key
#         print values.get('total_stock_value')


# for key, values in enron_data.items():
#     if key == ('Colwell Wesley'.upper()):
#         print key
#         print values.get('from_this_person_to_poi')


# for key, values in enron_data.items():
#     for name in ['LAY', 'SKILLING', 'FASTOW']:
#         if key.startswith(name):
#             print key, values.get('total_payments')


# no_quantified_salary = []
# no_known_email_add = []
# for key, values in enron_data.items():
#     no_quantified_salary.append(values.get('salary') != 'NaN')
#     no_known_email_add.append(values.get('email_address') != 'NaN')
#
# print (no_quantified_salary).count(True)
# print (no_known_email_add).count(True)

NaN_salary = []
for key, value in enron_data.items():
    NaN_salary.append(value.get('total_payments') == 'NaN' and value.get('poi') == 'True')

print NaN_salary
# print NaN_salary.count(True)*1.0 / len(enron_data)
