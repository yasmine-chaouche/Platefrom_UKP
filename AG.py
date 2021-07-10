import numpy as np
import time
import pandas as pd


def readData6(file_name):
    file_name_arr_str = file_name.split("\\", 3)
    type_instance = file_name_arr_str[1]
    print("Instance de type : " + type_instance)
    size_type = file_name_arr_str[2]

    title = file_name_arr_str[3]
    title_data = title.split("_")

    capacity = int(title_data[0].split("cap", 1)[1])
    instanceLength = int(title_data[1])
    print("Taille " + size_type + " : " + str(instanceLength))

    data = pd.read_csv(file_name)

    volumes = data["volumes"]
    profits = data["gains"]

    tab_data = []
    for volume, profit in zip(volumes, profits):
        tab_data.append([volume, profit])

    return capacity, instanceLength, tab_data


###############################################################################################

def get_max_number_item(items, capacity=0):
    tab_number = [capacity // item[0] for item in items]
    return tab_number, sum(tab_number)


def coisement_uniforme(parents, length):
    np.random.seed()
    tab_rand = np.random.rand(length)
    mask0 = tab_rand > 0.5
    mask1 = 1 - mask0
    fils = parents[0] * mask0 + parents[1] * mask1
    return fils


def generer_solutions(nb_solutions, taille, tab_poids, capacity):
    solutions = []
    for i in range(nb_solutions):
        np.random.seed()
        capacity_used = 0
        solution = np.zeros(taille, dtype=int)
        ind = np.random.randint(taille)
        while capacity_used + tab_poids[ind] < capacity:
            solution[ind] = 1
            capacity_used = capacity_used + tab_poids[ind]
            ind = np.random.randint(taille)
            while (solution[ind]):
                ind = np.random.randint(taille)
        solutions.append(solution)
    return solutions


def trier_objet_utility(items):
    items.sort(key=lambda x: x[1] / x[0], reverse=True)
    return items


def get_tab_gain_new(items_sorted, tab_max_nb):
    tab_gain = []
    for i in range(len(tab_max_nb)):
        tab = [items_sorted[i][1]] * tab_max_nb[i]
        tab_gain = tab_gain + tab
    return tab_gain


def get_tab_poid_new(items_sorted, tab_max_nb):
    tab_poid = []
    for i in range(len(tab_max_nb)):
        tab = [items_sorted[i][0]] * tab_max_nb[i]
        tab_poid = tab_poid + tab
    return tab_poid


def make_solution_realisable(solution, tab_poids, capacity):
    i = 0
    # print('fonction : make solution realisable')
    somme_poids = sum(np.array(solution) * np.array(tab_poids))
    while somme_poids > capacity and i < len(solution):

        # solution[-1-i]=0
        #  print("iteration i =",i,"taille du tableau",len(solution))
        if (solution[len(solution) - 1 - i] == 1):
            solution[len(solution) - 1 - i] = 0
            somme_poids = sum(np.array(solution) * np.array(tab_poids))
            # print("le poids de la solution courante est: ", somme_poids,
            #     "la capacité: ", capacity,"l'indice du changement de bit",len(solution)-1-i)
        i = i + 1
    i = 0

    # print("fin de la premiere phase de make solution realisable");
    # print("le poids de la solution réparée courante est: ", sum(np.array(solution) * np.array(tab_poids)), "la capacité: ",
    #   capacity)
    # print("debut 2eme boucle")
    somme_poids = sum(np.array(solution) * np.array(tab_poids))
    # print("somme_poids vaut",somme_poids)
    while i < len(solution) and somme_poids < capacity:
        if somme_poids + tab_poids[i] <= capacity:
            #       print("on ajoute l'item :",i)

            solution[i] = 1
            somme_poids = sum(np.array(solution) * np.array(tab_poids)) + tab_poids[i]

        #      print("nouveau poids occupé", somme_poids)
        i = i + 1

    # print("fin de la fonction make solution realisable")
    return (solution)


def mutation(solution, taille):
    np.random.seed()
    indice1 = np.random.randint(taille)
    while True:
        indice2 = np.random.randint(taille);
        if indice1 != indice2 or taille <= 1:
            break
    solution[indice1] = 1 - solution[indice1]
    solution[indice2] = 1 - solution[indice2]
    return solution


def eval_solution(solution, tab_gain_new):
    gain_total = sum(np.array(solution) * np.array(tab_gain_new))
    return gain_total


def solution_exist(sol, solutions):
    i = 0
    exist = False
    while (not exist) and i < len(solutions):
        if np.array_equal(sol, solutions[i]):
            exist = True
        i = i + 1
    return exist


def binaryToNsolution(solution, tab_max_nb):
    solN = []
    indMin = 0;
    for i in range(len(tab_max_nb)):
        indMax = indMin + tab_max_nb[i]
        solN.append(sum(solution[indMin:indMax]))
        indMin = indMax
    return solN


def get_pools_solutions(solutions, nb_per_group):
    taille = len(solutions)
    bool_sol = np.zeros(taille, dtype=int)
    i = 0;
    np.random.seed()
    ind = np.random.randint(taille)
    pool1 = []
    pool2 = []
    while i < 2 * nb_per_group:
        bool_sol[ind] = 1
        if i < nb_per_group:
            pool1.append(solutions[ind])
        else:
            pool2.append(solutions[ind])
        i = i + 1
        ind = np.random.randint(taille)
        while (bool_sol[ind]) and i < 2 * nb_per_group:
            ind = np.random.randint(taille)
    return pool1, pool2


def geneticAlgorithm(items, capacity, nb_tour, solutions, nb_solutions, nb_per_group, proba_mutation, pc):
    items_sorted = trier_objet_utility(items)
    # print('item sorted ==', items_sorted)
    tab_max_nb_items, taille = get_max_number_item(items_sorted, capacity)
    # print('tab_max_nb_items==',tab_max_nb_items)
    tab_poids_new = get_tab_poid_new(items_sorted, tab_max_nb_items)
    # print('la taille de tab_poids_new',len(tab_poids_new));
    # print('tab_poids_new==',tab_poids_new)
    tab_gain_new = get_tab_gain_new(items_sorted, tab_max_nb_items)
    # print('tab_gain_new==',tab_gain_new)
    #     solutions= generer_solutions(nb_solutions, taille,tab_poids_new, capacity)
    # print([binaryToNsolution(sol, tab_max_nb_items) for sol in solutions])
    evals = []
    # print(solutions)
    # print(np.array(solutions))
    for i in range(len(solutions)):
        evals.append(eval_solution(solutions[i], tab_gain_new))
    # print('evals==', evals)
    for k in range(nb_tour):
        # print('loop1=',k)
        ind_sol_best = np.argmax(evals)
        best_solution = solutions[ind_sol_best]
        loop = True;
        p = 0;
        while (loop):
            # print('loop2=',p)
            pool1, pool2 = get_pools_solutions(solutions, nb_per_group)
            evals1 = []
            evals2 = []

            for i in range(len(pool1)):
                evals1.append(eval_solution(pool1[i], tab_gain_new))

            for i in range(len(pool2)):
                evals2.append(eval_solution(pool2[i], tab_gain_new))

            best_sol_1 = pool1[np.argmax(evals1)]
            best_sol_2 = pool2[np.argmax(evals2)]

            np.random.seed()
            pp = np.random.rand()
            fils = coisement_uniforme([best_sol_1, best_sol_2], taille)

            np.random.seed()
            proba = np.random.rand()
            if proba < proba_mutation:
                fils = mutation(fils, taille)

            fils = make_solution_realisable(fils, tab_poids_new, capacity)
            #   print("apres le make solution realisable");
            if not solution_exist(fils, solutions):
                # print("la solution n'existe pas")
                loop = False;
            p = p + 1
        eval_fils = eval_solution(fils, tab_gain_new)
        ind_worst = np.argmin(evals)
        worst_sol = solutions[ind_worst]
        if eval_fils > evals[ind_worst]:
            solutions[ind_worst] = fils
            evals[ind_worst] = eval_fils
        bestNsol = binaryToNsolution(best_solution, tab_max_nb_items)
        gain_tot = eval_solution(best_solution, tab_gain_new)
        solution = []
        objects = []
        poids = 0
        for i, item in enumerate(bestNsol):
            if item != 0:
                objects.append(items[i])
                solution.append(item)
                poids += item * items[i][0]
    # print("le gain",gain_tot)
    # print('best solution==', bestNsol)
    return objects, solution, gain_tot, poids