#!/usr/bin/python
# coding=utf-8


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []
    err = []
    temp = []
    ### your code goes here
    for i in range(len(predictions)):
        # err = numpy.fabs(net_worths - predicitons)
        err.append(abs(net_worths[i] - predictions[i]))
        temp.append((ages[i], net_worths[i], err[i]))

    # 从小到大排序
    err.sort()
    err = err[:81]


    for j in range(len(temp)):
        if temp[j][2] in err:
            cleaned_data.append(temp[j])

    return cleaned_data

