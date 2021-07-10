import pandas as pd
import sys
from timeit import default_timer as timer
from datetime import time, timedelta
# la classe qui contient les infos de notre instance ukp
class ukp:
    capacity = 0
    p = []  # Profits Array
    w = []  # Weights Array

    #constructeur de la classe ukp
    def __init__(self, capacity, p, w):
        self.capacity = capacity
        self.p = list(p)
        self.w = list(w)
    #méthode pour valider les valeurs

# la solution ukp ui contient le total profit
# la liste des objets pris
# et combien de fois on a pris l'objet

class ukp_solution:
    def __init__(self):
        self.total = 0 # profit total
        self.tpoids=0  # total poids des objets pris
        self.taken = [] #liste des objets
        self.fois = [] #chaque objets combien d'exemplaire on a pris

# cette méthode insert un objet dans la structure de la solution
def ukp_select_object (_solution, objet):
        if not objet in _solution.taken:
            _solution.taken.append(objet)
            _solution.fois.insert(objet, 1)
        else :
            _solution.fois[0] += 1

# Weight-Ordered heuristic
def weight_heuristic(objet):

    cap = objet.capacity # capacité restante pour chaque itération
    temp = list(objet.w)
    index = list(range(0, len(objet.p)))  # index des objets trié
    #on instancie la solution
    ukp_solotion= ukp_solution()
    #ordonner l'index du tableau
    #organiser l'index
    for iter in range(len(objet.w) - 1, 0, -1):
        for idx in range(iter):
            if temp[idx] > temp[idx + 1]:
                temp[idx], temp[idx + 1] = temp[idx+1], temp[idx]
                index[idx], index[idx + 1] = index[idx+1], index[idx]
    i = 0
    # on commence à remplir le sac et
    # on retranche la capacité à chaque itération
    # et on garde trace de profits aussi
    while cap > 0 and i < len(objet.w):
        #si on a encore de place
        if objet.w[index[i]] < cap:
            ukp_select_object(ukp_solotion, index[i])
            ukp_solotion.total += objet.p[index[i]]
            ukp_solotion.tpoids += objet.w[index[i]]
            cap -= objet.w[index[i]]
            #print(ukp_solotion.tw)
        # sinon on avance dans la liste des objets disponible
        else:
            i = i+1
        # on retourne un objet de type ukp qui contient notre solution
    return ukp_solotion


# méthode qui fait appel à l'heuristique et lui donne l'instance en entrée
def test_methode(methode, instance_ukp):
    # point de début de l'execution on garde le temps
    debut = timer()
    if methode == "weight_heuristic":
        solution = weight_heuristic(instance_ukp)
        # fin d'execution on garde le temps
    temps = str(timedelta(seconds=timer() - debut))
    return temps
import csv

class Item :
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value
    def set_occurance(self,c):
        self.occu = c
    def get_occurance(self):
        return self.occu
    def __str__(self):
        return "weight : "+str(self.weight)+" value : "+str(self.value)

def readData(file_name):
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