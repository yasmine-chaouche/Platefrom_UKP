import sys
import time
from timeit import default_timer as timer, main
from datetime import time, timedelta
import pandas as pd

# This is the class that contains the information
# of an instance : capacity, weights list and profits list


class ukp:
    capacity = 0
    p = []  # Profits Array
    w = []  # Weights Array

    def __init__(self, capacity, p, w):
        self.capacity = capacity
        self.p = list(p)
        self.w = list(w)


# This is useful when you need a pair of
# (value, object) in your algorithm
class rowtwin:
    def __init__(self, obj, row):
        self.row = row
        self.obj = obj

# This is what all the algorithms must return
# the total profit, the list of taken objects
# and the list that contains how many times
# each object has been taken
class ukp_solution:
    def __init__(self):
        self.total = 0	
        self.tpoids = 0 
        self.taken = []
        self.ttimes = []

# This method inserts an object into the solutions structure
def ukp_select_object (ukp_solution_o, object):
        if not object in ukp_solution_o.taken:
            ukp_solution_o.taken.append(object)
            ukp_solution_o.ttimes.insert(object, 1)
        else :
            ukp_solution_o.ttimes[object] += 1


# Density ordered heuristic, it takes a ukp object as parameter
# and return a ukp_solution
def ukp_density(ukp_obj):
    twins = []

    for i in range(0, len(ukp_obj.p)):
        tmp_row = int(ukp_obj.p[i]/ukp_obj.w[i])
        twins.append(rowtwin(i, tmp_row))

    twins.sort(key=lambda x: x.row, reverse=True)
    # for twin in twins:
    #     print (str(twin.obj) + ":" + str(twin.row))

    current_capacity = ukp_obj.capacity
    total_profit = 0
    current_obj_i = 0
    current_obj = twins[current_obj_i].obj
    cont = True

    ukp_sol_o = ukp_solution()

    while current_capacity > 0 and cont:
        if (ukp_obj.w[current_obj] > current_capacity):
            if ++current_obj_i < len(ukp_obj.p):
                current_obj = twins[current_obj_i].obj
            else:
                cont = False
            continue

        ukp_select_object(ukp_sol_o, current_obj)
        ukp_sol_o.total += ukp_obj.p[current_obj]
        current_capacity -= ukp_obj.w[current_obj]

    return ukp_sol_o

# total value heuristic
def ukp_tv(ukp_object):
    cc = ukp_object.capacity # left capacity in each iteration

    rem_objects = list(range(0, len(ukp_object.p))) # Remaining objects
    ukp_sol_o = ukp_solution()

    while (cc > 0 and len(rem_objects) > 0):
        max_metric = 0; selected = -1
        for object in rem_objects:
            metric = ukp_object.p[object] * int(cc/ukp_object.w[object])
            if metric > max_metric:
                max_metric = metric
                selected = object

        cc = cc - ukp_object.w[selected]
        if cc < 0:
            break

        ukp_select_object(ukp_sol_o, selected)
        ukp_sol_o.tpoids += ukp_object.w[selected]
        rem_objects.remove(selected)

    return ukp_sol_o


def execute_instance(type, ukp_o):
    # start = time.localtime()
    debut = timer()
    if type == "ukp_density":
        solution = ukp_density(ukp_o)
    elif type == "ukp_tv":
        solution = ukp_tv(ukp_o)
    else :
        return
    temps = str(timedelta(seconds=timer() - debut))
    #end = time.localtime()

    #print (type)
    #print ("Executed in " + temps + " secs")
	#print ("Executed in " + str(end.tm_sec - start.tm_sec) + " secs")
    #print ("Total profit = " + str(solution.total))
	#print ("Total weight = " + str(solution.tpoids))
    #print ("chosen objects:times")
    #for i in range(0, len(solution.taken)):
        #print(str(solution.taken[i]) + ":" + str(solution.ttimes[i]))


# main


# '1000_591952.csv'
#a=['cap591952_5000_facile.csv']
#tmp=[]
#instances=[]
#for g in range (len(a)):
#tps1 = time.clock()
#data=read_csv('cap591952_5000_facile.csv')
#wt=data['volumes'].tolist()
#val=data['gains'].tolist()
#w=a[g].split('_')
#h=w[1].split('.')
#W=int(h[0]) #la capacité
#n = len(val)
#instances.append(n)
#elements=[]
#listpoids=[]
#listvaleurs=[]


#sol=unboundedKnapsac­k_DP2(W, n, val, wt)
#elements=backtrackin­g(sol,W, n, val, wt)




#for i in range (len(elements)):
#listpoids.append(wt[­elements[i]])
#listvaleurs.append(v­al[elements[i]])


# print("Solution optimale:",sol[W])
# print("Items:",eleme­nts)
# print("Poids des items:",listpoids)
# print("Valeurs des items:",listvaleurs)
#tps2 = time.clock()
#tmp.append(tps2-tps1­)

#print(instances)
#print(tmp)

#import matplotlib.pyplot as plt
#import numpy as np

#plt.figure(figsize=(­10,8), dpi=80)
#plt.plot(instances,t­mp, color = "forestgreen")
#plt.show()

#instance = ukp(10, [6, 6, 7, 8], [1, 2, 3, 4])
#instance = ukp(100000, wt, val)

import csv 
class Item :
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value
    def set_occurance(self,c):
        self.occu = c
    def get_occurance() :
        return self.occu
    def __str__(self):
        return "weight : "+str(self.weight)+" value : "+str(self.value)

    # function to read the data
def readData2(file_name):
    file_name_arr_str = file_name.split("/",3)
    type_instance = file_name_arr_str[1]
    print("Instance de type : "+type_instance)
    size_type = file_name_arr_str[2]
    
    title = file_name_arr_str[3]
    title_data = title.split("_")

    capacity = int(title_data[0].split("cap",1)[1])
    instanceLength = int(title_data[1])
    print("Taille "+ size_type+" : "+str(instanceLength))

    data = []
    with open(file_name) as datasetFile :
        csv_reader = csv.reader(datasetFile,delimiter=",")
        line_count = 0
        for row in csv_reader:
            # columns name : 
            # volumes,gains
            if(line_count != 0 ):
                data.append(Item(int(row[0]),int(row[1])))
            line_count = line_count + 1
    return capacity , instanceLength , data

#listes des paths des fichiers 
a=["Datasets/Facile/Moyenne/cap591952_5000_facile.csv","Datasets/Difficile/Moyenne/cap1596642_5000_diff.csv","Datasets/Difficile/Grande/cap1596642_10000_diff.csv",
"Datasets/Facile/Grande/cap7547243_10000_facile.csv",
"Datasets/Moyenne/Grande/cap52926330_10000_moy.csv","Datasets/Moyenne/Moyenne/cap3897377_5000_moy.csv"]
tmps =[] #les temps d'execution 
inst = [] #les capacités des instances
x=0 
for x in range(len(a)) :

        instance=readData2(a[x])

        #print("capacité du sac "+str(instance[0]))

        capp=instance[0]
		
        inst.append(capp)
        listspoids=[]
        listsprofits=[]
        dat=instance[2] #les objets sont ici 
        
        j=0
        for j in range(len(dat)):
            listspoids.append(dat[j].weight)
            listsprofits.append(dat[j].value)
        
        instances = ukp(capp,listspoids,listsprofits)
        tmps.append(execute_instance("ukp_tv", instances))


# tracé du graphe 
#import matplotlib.pyplot as plt
#import numpy as np

#plt.figure(dpi=80)
#plt.plot(inst,tmps, color = "forestgreen")
#plt.show()

#execute_instance("ukp_dno", instance)
#print ("\n")
#execute_instance("ukp_tv", instance)


def readData3(file_name):
    file_name_arr_str = file_name.split("\\", 3)
    type_instance = file_name_arr_str[1]
    #print("Instance de type : " + type_instance)
    size_type = file_name_arr_str[2]

    title = file_name_arr_str[3]
    title_data = title.split("_")

    capacity = int(title_data[0].split("cap", 1)[1])
    instanceLength = int(title_data[1])
    #print("Taille " + size_type + " : " + str(instanceLength))

    data =pd.read_csv(file_name)
    return capacity, instanceLength, data