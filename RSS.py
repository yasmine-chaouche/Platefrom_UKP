import math
import numpy as np
# import Greedy
# from getData import getData
import time


# Created by Massina feat Amine
def trier_objet_utility(items):
    items.sort(key=lambda x: x[1] / x[0], reverse=True)
    return items


def get_tab_gain_new(items_sorted, tab_max_nb):
    tab_gain = []
    for i in range(len(tab_max_nb)):
        tab = [items_sorted[i][1]] * tab_max_nb[i]
        tab_gain = tab_gain + tab
        # print('tab_gain : ',tab_gain)
    return tab_gain


def get_tab_poid_new(items_sorted, tab_max_nb):
    tab_poid = []
    for i in range(len(tab_max_nb)):
        tab = [items_sorted[i][0]] * tab_max_nb[i]
        tab_poid = tab_poid + tab
    return tab_poid


def eval_solution(solution, tab_gain_new):
    # gain_total= sum(np.array(solution)* np.array(tab_gain_new))
    gain_total = 0
    for i in range(len(solution)):
        gain_total = gain_total + solution[i] * tab_gain_new[i]
    return gain_total


def get_max_number_item(items, capacity=0):
    tab_number = [capacity // item[0] for item in items]
    return tab_number, sum(tab_number)


def get_poids_total(bsol, tab_poid_new):
    poid_total = 0
    for i in range(len(bsol)):
        poid_total = poid_total + bsol[i] * tab_poid_new[i]
    return poid_total


def ntobinary(nsol, max_num_tab):
    bsol = []
    for i in range(len(max_num_tab)):
        for p in range(nsol[i]):
            bsol.append(1)
        for p in range(nsol[i], max_num_tab[i]):
            bsol.append(0)
    return bsol


def binaryToNsolution(solution, tab_max_nb):
    solN = []
    indMin = 0
    for i in range(len(tab_max_nb)):
        indMax = indMin + tab_max_nb[i]
        solN.append(sum(solution[indMin:indMax]))
        indMin = indMax
    return solN


def cool(temprature, coolingFactor):
    return temprature * coolingFactor


def getNeighbour(solution, taille, tab_poids_new, capacity):
    np.random.seed()
    sol = solution.copy()
    i = 0;
    x = np.random.randint(taille)
    if sol[x] == 1:
        sol[x] = 0
    else:
        capacityRest = capacity - get_poids_total(sol, tab_poids_new)
        listItemCanEnter = []
        for i in range(len(sol)):
            if capacityRest > tab_poids_new[i] and sol[i] == 0:
                listItemCanEnter.append(i)
        if len(listItemCanEnter) != 0:
            ind = np.random.randint(len(listItemCanEnter))
            sol[listItemCanEnter[ind]] = 1
        else:
            listItemPris = []
            for i in range(len(sol)):
                if sol[i] == 1:
                    listItemPris.append(i)
            if len(listItemPris) != 0:
                ind = np.random.randint(len(listItemPris))
                sol[listItemPris[ind]] = 0
    return sol


def getNextState(solution, taille, tab_poids_new, tab_gain_new, capacity, temperature):
    newSolution = getNeighbour(solution, taille, tab_poids_new, capacity);
    evalNewSol = eval_solution(newSolution, tab_gain_new)
    evalOldSol = eval_solution(solution, tab_gain_new)
    delta = evalNewSol - evalOldSol
    if (delta > 0):
        return newSolution
    else:
        x = np.random.rand()
        if (x < math.exp(delta / temperature)):
            return newSolution
        else:
            return solution


def simulatedAnnealing(itemsIn, capacity, solinit, samplingSize, temperatureInit, coolingFactor, endingTemperature):
    items = itemsIn.copy()
    for i in range(len(items)):
        items[i].append(solinit[i])
    items_sorted = trier_objet_utility(items)
    # print(items_sorted)
    solinitsorted = []
    for i in range(len(items_sorted)):
        solinitsorted.append(items_sorted[i][2])
    # solinitsorted = solinit.copy()
    # ♥print('solution n sorted',solinitsorted)
    tab_max_nb, taille = get_max_number_item(items_sorted, capacity)
    tab_poids_new = get_tab_poid_new(items_sorted, tab_max_nb)
    tab_gain_new = get_tab_gain_new(items_sorted, tab_max_nb)

    # print('tab_max_nb',tab_max_nb)
    # print('tab_gain_new',tab_gain_new)
    # print('tab_poids_new',tab_poids_new)
    solCurrent = ntobinary(solinitsorted, tab_max_nb)
    # print('le tableau du gain new est \t ',tab_gain_new)
    evalsol = eval_solution(solCurrent, tab_gain_new)
    # print('evaluation de la solution initiale du RS \t  ',evalsol)
    temperature = temperatureInit
    bestSol = solCurrent.copy()
    bestEval = evalsol

    # print('best first sol',bestSol)
    # print('eval best sol', bestEval)
    while (temperature > endingTemperature):
        # print('boucle dans le while')
        for i in range(samplingSize):
            solCurrent = getNextState(solCurrent, taille, tab_poids_new, tab_gain_new, capacity, temperature)
            # print('récupere un voisn')
            evalCurrent = eval_solution(solCurrent, tab_gain_new);
            # print('current_sol',solCurrent,binaryToNsolution(solCurrent,tab_max_nb),evalCurrent, 'best eval',bestEval, bestSol)

            if evalCurrent > bestEval:
                bestSol = solCurrent.copy()
                bestEval = evalCurrent
                # print("remplacement pas une meilleur sol")
        temperature = cool(temperature, coolingFactor)
    # print(bestSol)
    objects = []
    solution = []
    Nsol = binaryToNsolution(bestSol, tab_max_nb)
    for i, item in enumerate(Nsol):
        if item != 0:
            objects.append(items[i])
            solution.append(item)
    poids = 0
    for i, obj in enumerate(objects):
        poids += obj[0] * solution[i]

    return objects, solution, Nsol, bestEval, poids


# test
# items= [[3, 10], [5, 3],[3,5]]1
# items= [[2,5],[3,2],[5,10],[7,20]]

# capacity=13
# nb=len(items)
# capacity = 13
# solinit=[2,0,1]
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