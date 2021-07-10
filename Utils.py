import math
import numpy as np
import time
import csv
import random


class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

    def set_occurance(self, c):
        self.occu = c

    def get_occurance():
        return self.occu

    def __str__(self):
        return "weight : " + str(self.weight) + " value : " + str(self.value)


# function to read the data
def readData11(file_name):
    file_name_arr_str = file_name.split("/", 3)
    type_instance = file_name_arr_str[1]
    print("Instance de type : " + type_instance)
    size_type = file_name_arr_str[2]

    title = file_name_arr_str[3]
    title_data = title.split("_")

    capacity = int(title_data[0].split("cap", 1)[1])
    instanceLength = int(title_data[1])
    print("Taille " + size_type + " : " + str(instanceLength))

    data = []
    with open(file_name) as datasetFile:
        csv_reader = csv.reader(datasetFile, delimiter=",")
        line_count = 0
        for row in csv_reader:
            # columns name :
            # volumes,gains
            if (line_count != 0):
                data.append(Item(int(row[0]), int(row[1])))
            line_count = line_count + 1
    return capacity, instanceLength, data


def arrayOf(items):
    array = []
    for item in items:
        array.append([item.weight, item.value])
    return array


# item.value/item.weight
def sort_items_by_profit(items):
    items.sort(key=lambda item: item[1] / item[0], reverse=True)


def gen_random_sol(tab, n, capacity):
    weight = []
    profits = []
    capacityleft = capacity
    sol = []
    # gain=0
    for k in range(0, n):
        sol.append(0)
    for i in range(0, n):
        weight.append(tab[i][0])
        profits.append(tab[i][1])
    j = 0
    while (j < n and capacityleft > 0):
        index = np.random.randint(0, n - 1)
        maxQuantity = int(capacityleft / weight[index]) + 1
        if (maxQuantity == 0):
            nbItems = 0
        else:
            nbItems = np.random.randint(0, maxQuantity)
        sol[index] = nbItems
        capacityleft = capacityleft - weight[index] * sol[index]

        # gain= gain + profits[index]*sol[index]
        j = j + 1
    gain_out = 0
    for i in range(n):
        gain_out = gain_out + profits[i] * sol[i]

    return gain_out, capacityleft, sol