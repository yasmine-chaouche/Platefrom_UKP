
import streamlit as st
import altair as alt
import os
import glob
import pandas as pd
import time
from DP import unboundedKnapsack_DP2
from DP import backtracking
from WOH import ukp,weight_heuristic,readData,test_methode
import matplotlib.pyplot as plt
from PIL import Image
from tv import ukp, ukp_tv, execute_instance, readData2, readData3
from AG import *
from RS import*
from hyper_ga import Hyper_GA

from BranchAndBound import knapsack_BB ,unboundedKnapsack
import plotly.express as px

def csv_to_items(filename):
    capacity = filename.split('/')[-1].split('_')[-1].split('.')[0]
    data = pd.read_csv(filename)
    volumes =data['volumes'].tolist()
    gains =data['gains'].tolist()
    items = list()
    for item in range(0,len(volumes)):items.append([volumes[item] ,gains[item]])
    return items ,int(capacity)

def main():
    p1=[" " ,"Branch & Bound", "Programmation dynamique"]
    default1=p1.index(" ")
    p2=[" " ,"Weight Ordered Heuristique","Total Value Heuristique" ]
    default2= p2.index(" ")
    p3=[" " ,"Algorithme Génétique", "Récuit Simulé"]
    default3=p3.index(" ")
    p4=[" " ,"hyper-heuristique AG de AG"]
    default4=p4.index(" ")

    pj = st.sidebar.button("HOME")
    page1 = st.sidebar.selectbox("Choisir une méthode exacte", p1,index=0)
    page2 = st.sidebar.selectbox("Choisir une heuristique",p2,index=0)
    page3 = st.sidebar.selectbox("Choisir une méthaheuristique", p3,index=0)
    page4 = st.sidebar.selectbox("Choisir une hyper-heuristique", p4,index=0)

    if pj:
        st.title("WELCOME TO OUR APPLICATION")
        image = Image.open('EN.png')
        image2 = Image.open('knapsack.png')
        st.image(image)
        st.image(image2)


    if (page1==" "):
        st.title(" ")

    elif (page1 == "Branch & Bound" ):
        st.title("Branch and Bound")
        st.write("entrer manuellement les données:")
        poids,gain,capa=st.beta_columns([2, 2,2])
        poids = poids.text_input("volume")
        gain = gain.text_input("gain")
        capa=capa.number_input("capacité")
        capa=int(capa)

        if st.button("Add row"):
            get_data().append({"volumes": poids, "gains": gain})

        df = pd.DataFrame(get_data())
        st.write(df)

        if st.button("Add csv file"):
            df.to_csv('myFile.csv', columns=["volumes", "gains"], index=False)


        #################################
        filename = file_selector()
        st.write('Vous avez choisi le fichier: `%s`' % filename)
        data = pd.read_csv(filename)
        st.write(data)
        # print(data)

        data = pd.read_csv(filename)
        wt = data['volumes'].tolist()
        val = data['gains'].tolist()

        a = filename.split('\\')[-1].split('_')[0]
        W = int(filename.split('\\')[-1].split('_')[1].split('.')[0])  # la capacité
        n = int(filename.split('\\')[-1].split('_')[0])

        if (filename=="D:\\optim_project\\myFile.csv"):
            W=capa

        if (n > 65):
            start = time.time()
            unboundedKnapsack(W, n, val, wt)
            end = time.time()
            st.write("le poid du sack est: ", W)
            st.write("le gain maximal est: ", unboundedKnapsack(W, n, val, wt))
            st.write("le temps de calcul est: ", (end - start) * 1000, "ms")
        else:
            items, capacity = csv_to_items(filename)
            sol = knapsack_BB(items, capacity)
            start = time.time()
            unboundedKnapsack(W, n, val, wt)
            end = time.time()
            st.write("le poid du sack est: ", sol[1])
            st.write("le gain maximal est: ", sol[0])
            for elem in sol[2]:
                st.write(elem[1], " de l'element avec le volume: ", elem[0])
            st.write("le temps de calcul est: ", (end - start) * 1000, "ms")


        st.title("Instances de taille entre 5 et 205 objets ")
        if(st.button("AFFICHER LE GRAPHE")):
            image = Image.open('BB.png')
            st.image(image)




    elif (page1 == "Programmation dynamique" ):
        st.title("Programmation Dynamique DP")
        st.write("entrer manuellement les données:")
        poids,gain,capa=st.beta_columns([2, 2,2])
        poids = poids.text_input("volume")
        gain = gain.text_input("gain")
        capa=capa.number_input("capacité")
        capa=int(capa)

        if st.button("Add row"):
            get_data().append({"volumes": poids, "gains": gain})

        df = pd.DataFrame(get_data())
        st.write(df)

        if st.button("Add csv file"):
            df.to_csv('myFile.csv', columns=["volumes", "gains"], index=False)


        ########################################################################
        filename = file_selector()
        st.write('Vous avez choisi le fichier: `%s`' % filename)
        data = pd.read_csv(filename)
        st.write(data)
        tps3 = time.clock()
        wt = data['volumes'].tolist()
        val = data['gains'].tolist()
        if (filename=="D:\\optim_project\\myFile.csv"):
            W=capa
        else:
            a = filename.split('\\')
            d = a[2].split('_')
            h = d[1].split('.')
            W = int(h[0])  # la capacité

        st.write("la capacité",W)

        n = len(val)
        elements = []
        listpoids = []
        listvaleurs = []

        sol = unboundedKnapsack_DP2(W, n, val, wt)
        elements = backtracking(sol, W, n, val, wt)

        for i in range(len(elements)):
            listpoids.append(wt[elements[i]])
            listvaleurs.append(val[elements[i]])

        st.write("Solution optimale:",sol[W])
        st.write("Poids total: ",sum(listpoids))
        st.write("Items:",elements)
        st.write("Poids des items:", listpoids)
        st.write("Valeurs des items:", listvaleurs)
        tps4 = time.clock()
        st.write("Execution time:", tps4 - tps3)

        st.title("Instances de taille entre 5 et 205 objets ")
        if(st.button("Afficher graphe")):

            # '1000_591952.csv'
            a = ['205_129.csv', '200_180.csv', '195_410.csv', '190_841.csv', '185_659.csv', '180_997.csv', '175_523.csv',
                 '170_951.csv', '165_703.csv', '160_580.csv', '155_752.csv', '150_143.csv', '145_531.csv', '140_208.csv',
                 '135_578.csv', '130_969.csv', '125_765.csv', '120_371.csv', '115_777.csv', '110_571.csv', '105_436.csv',
                 '100_468.csv', '95_629.csv', '90_859.csv', '85_850.csv', '80_348.csv', '75_618.csv', '70_510.csv',
                 '65_666.csv', '60_655.csv', '55_704.csv', '50_490.csv', '45_788.csv', '40_362.csv', '35_247.csv',
                 '30_493.csv', '25_484.csv', '20_718.csv', '15_216.csv', '10_154.csv']
            tmp = []
            instances = []
            for g in range(len(a)):
                tps1 = time.clock()
                data = pd.read_csv(a[g])
                wt = data['volumes'].tolist()
                val = data['gains'].tolist()
                w = a[g].split('_')
                h = w[1].split('.')
                W = int(h[0])  # la capacité
                n = len(val)
                instances.append(n)
                elements = []
                listpoids = []
                listvaleurs = []

                sol = unboundedKnapsack_DP2(W, n, val, wt)
                elements = backtracking(sol, W, n, val, wt)

                for i in range(len(elements)):
                    listpoids.append(wt[elements[i]])
                    listvaleurs.append(val[elements[i]])

                #     print("Solution optimale:",sol[W])
                #     print("Items:",elements)
                #     print("Poids des items:",listpoids)
                #     print("Valeurs des items:",listvaleurs)
                tps2 = time.clock()
                tmp.append(tps2 - tps1)

            #st.write("tailles des instaces",instances)
            #st.write("temps d'exécution",tmp)

            fig=plt.figure(figsize=(10, 8), dpi=80)
            plt.plot(instances, tmp, color="forestgreen", label="DP")
            plt.xlabel("taille des instances (objets)")
            plt.ylabel("temps d'exécution (s)")
            plt.legend()
            st.pyplot(fig)

    if page2== "Weight Ordered Heuristique":
        st.title("Weight Ordered Heuristique")
        st.write("entrer manuellement les données:")
        poids,gain,capa=st.beta_columns([2, 2,2])
        poids = poids.text_input("volume")
        gain = gain.text_input("gain")
        capa=capa.number_input("capacité")
        capa=int(capa)

        if st.button("Add row"):
            get_data().append({"volumes": poids, "gains": gain})

        df = pd.DataFrame(get_data())
        st.write(df)

        if st.button("Add csv file"):
            df.to_csv('D:\\optim_project\\Datasets\\myFile.csv', columns=["volumes", "gains"], index=False)

            ###########################################################
        filename = file_selector2()
        st.write('Vous avez choisi le fichier: `%s`' % filename)

        capacity, instanceLength, data=readData(filename)

        wt = data['volumes'].tolist()
        val = data['gains'].tolist()

        tps5 = time.clock()

        capacity=int(capacity)
        st.write("Capacité :",capacity)
        st.write("Taille d'instance :",instanceLength)
        instance=ukp(capacity,val,wt)
        solution=weight_heuristic(instance)

        tps6 = time.clock()
        st.write("Temps d'Execution:", tps6 - tps5)
        ###########
        st.write(" Le Total profit = ",solution.total)
        st.write(" Le Total weight = ",solution.tpoids)
        st.write("Items pris :",solution.taken," Nombre de fois : " ,solution.fois)
        ###################
        #graphe
        st.title("Fichier de taille entre 5000 et 10000 objets ")
        if(st.button("AFFICHER LE GRAPHE")):
            image = Image.open('WOH.PNG')
            st.image(image)






    elif page2=="Total Value Heuristique":
        st.title("Page Total Value Heuristique")
        st.write("entrer manuellement les données:")
        poids, gain, capa = st.beta_columns([2, 2, 2])
        poids = poids.text_input("volume")
        gain = gain.text_input("gain")
        capa = capa.number_input("capacité")
        capa = int(capa)

        if st.button("Add row"):
            get_data().append({"volumes": poids, "gains": gain})

        df = pd.DataFrame(get_data())
        st.write(df)

        if st.button("Add csv file"):
            df.to_csv('D:\\optim_project\\Datasets\\myFile.csv', columns=["volumes", "gains"], index=False)

            ###########################################################
        filename = file_selector2()
        st.write('Vous avez choisi le fichier: `%s`' % filename)

        capacity, instanceLength, data = readData3(filename)

        wt = data['volumes'].tolist()
        val = data['gains'].tolist()

        tps7 = time.time()

        capacity = int(capacity)
        st.write("Capacité :", capacity)
        st.write("Taille d'instance :", instanceLength)
        instance = ukp(capacity, val, wt)
        solution = ukp_tv(instance)

        tps8 = time.time()
        st.write("Temps d'Execution:", tps8 - tps7)
        ###########
        # st.write(" Le Total profit = ",solution.total)
        st.write(" Le Total weight = ", solution.tpoids)
        st.write("Items pris :", solution.taken, " Nombre de fois : ", solution.ttimes)
        ###################
        # graphe
        st.title("Fichier de taille entre 5000 et 10000 objets ")
        # a = ["Datasets\\Facile\\Moyenne\\cap591952_5000_facile.csv","Datasets\\Facile\\Grande\\cap7547243_10000_facile.csv",
        #     "Datasets\\Moyenne\\Moyenne\\cap3897377_5000_moy.csv", "Datasets\\Moyenne\\Grande\\cap52926330_10000_moy.csv",
        #    "Datasets\\Difficile\\Moyenne\\cap1596642_5000_diff.csv","Datasets\\Difficile\\Grande\\cap1596642_10000_diff.csv"]
        a = ["Datasets/Facile/Moyenne/cap591952_5000_facile.csv", "Datasets/Difficile/Moyenne/cap1596642_5000_diff.csv",
             "Datasets/Difficile/Grande/cap1596642_10000_diff.csv",
             "Datasets/Facile/Grande/cap7547243_10000_facile.csv",
             "Datasets/Moyenne/Grande/cap52926330_10000_moy.csv", "Datasets/Moyenne/Moyenne/cap3897377_5000_moy.csv"]
        tmpp = []
        instancess = []

        for g in range(len(a)):

            st.write("C'est le fichier :", a[g])
            instance = readData2(a[g])
            # instance=readData2(a[g])
            dat = instance[2]  # les objets sont ici
            capacity = instance[0]
            instanceLength = instance[1]
            # instanceLength=instance[1]
            listspoids = []
            listsprofits = []

            j = 0
            for j in range(len(dat)):
                listspoids.append(dat[j].weight)
                listsprofits.append(dat[j].value)

            tps9 = time.time()
            capacity = int(capacity)
            # capacity = intance[0]
            st.write("Capacité :", capacity)
            st.write("Taille d'instance :", instanceLength)
            instance = ukp(capacity, listspoids, listsprofits)
            solution = ukp_tv(instance)
            # st.write("Cout total :",solution.total)
            tps10 = time.time()
            n = len(listsprofits)
            instancess.append(n)
            tmpp.append(tps10 - tps9)

        fig = plt.figure(figsize=(10, 8), dpi=80)
        plt.plot(instancess, tmpp, color="forestgreen", label="TVH")
        plt.xlabel("taille des instances (objets)")
        plt.ylabel("temps d'exécution (s)")
        plt.legend()
        st.pyplot(fig)

    elif page3=="Algorithme Génétique":
        st.header("Algorithme Génétique")
        st.write("entrer manuellement les données:")
        poids,gain,capa=st.beta_columns([2, 2,2])
        poids = poids.text_input("volume")
        gain = gain.text_input("gain")
        capa=capa.number_input("capacité")
        capa=int(capa)

        if st.button("Add row"):
            get_data().append({"volumes": poids, "gains": gain})

        df = pd.DataFrame(get_data())
        st.write(df)

        if st.button("Add csv file"):
            df.to_csv('myFile.csv', columns=["volumes", "gains"], index=False)

        ########################################################################
        filename = file_selector2()
        st.write('Vous avez choisi le fichier: `%s`' % filename)

        cap, nb, items = readData6(filename)

        ############################################################################""
        start = time.time()
        nb_solutions = 100
        items_sorted = trier_objet_utility(items)
        tab_max_nb_items, taille = get_max_number_item(items_sorted, cap)
        tab_poids_new = get_tab_poid_new(items_sorted, tab_max_nb_items)
        tab_gain_new = get_tab_gain_new(items_sorted, tab_max_nb_items)
        solutions = generer_solutions(nb_solutions, taille, tab_poids_new, cap)
        objects, solution, gain_tot, poids = geneticAlgorithm(items, cap, 20, solutions, nb_solutions, 2, 0.05, 0.8)
        end = time.time()
        st.write("le gain: ", gain_tot)
        st.write( "Temps d'éxécution : ", end - start)
        ################################################################################"""
        st.title("Fichier de taille entre 5000 et 10000 objets ")
        if(st.button("afficher graphe")):
            aa = ["Datasets\\Facile\\Moyenne\\cap591952_5000_facile.csv",
                  # "Datasets\\Facile\\Grande\\cap7547243_10000_facile.csv",
                  "Datasets\\Moyenne\\Moyenne\\cap3897377_5000_moy.csv",
                  "Datasets\\Moyenne\\Grande\\cap52926330_10000_moy.csv",
                  "Datasets\\Difficile\\Moyenne\\cap1596642_5000_diff.csv",
                  "Datasets\\Difficile\\Grande\\cap1596642_10000_diff.csv"]

            tmpp = []
            instancess = []
            for g in range(len(aa)):
                st.write("ggg", aa[g])
                cap, nb, items = readData6(aa[g])

                # wt = items[0].tolist()
                # val = items[1].tolist()

                start = time.time()
                nb_solutions = 100
                items_sorted = trier_objet_utility(items)
                tab_max_nb_items, taille = get_max_number_item(items_sorted, cap)
                tab_poids_new = get_tab_poid_new(items_sorted, tab_max_nb_items)
                tab_gain_new = get_tab_gain_new(items_sorted, tab_max_nb_items)
                solutions = generer_solutions(nb_solutions, taille, tab_poids_new, cap)
                objects, solution, gain_tot, poids = geneticAlgorithm(items, cap, 20, solutions, nb_solutions, 2, 0.05, 0.8)
                end = time.time()
                n = len(items[1])
                instancess.append(n)
                tmpp.append(end - start)

            fig = plt.figure(figsize=(10, 8), dpi=80)
            plt.plot(instancess, tmpp, color="forestgreen", label="WOH")
            #plt.px.bar(mean_counts_by_hour, x='hour', y='count', color='season', height=400)
            plt.xlabel("taille des instances (objets)")
            plt.ylabel("temps d'exécution (s)")
            plt.legend()
            st.pyplot(fig)









    elif page3=="Récuit Simulé":
        st.header("Récuit simulé")
        st.write("entrer manuellement les données:")
        poids,gain,capa,temp,choix=st.beta_columns([2, 2,2,2,2])
        poids = poids.text_input("volume")
        gain = gain.text_input("gain")
        capa=capa.number_input("capacité")
        temp = temp.text_input("température")
        choix = st.radio("Solution initiale aléatoire: ", ('OUI', 'NON'))

        # conditional statement to print
        # Male if male is selected else print female
        # show the result using the success function
        if (choix == 'Male'):
            st.success("Male")

        capa=int(capa)

        if st.button("Add row"):
            get_data().append({"volumes": poids, "gains": gain})

        df = pd.DataFrame(get_data())
        st.write(df)

        if st.button("Add csv file"):
            df.to_csv('myFile.csv', columns=["volumes", "gains"], index=False)

        ########################################################################
        filename = file_selector2()
        st.write('Vous avez choisi le fichier: `%s`' % filename)

        cap, nb, items = readData6(filename)

        ############################################################################""
        start = time.time()
        nb_solutions = 100
        items_sorted = trier_objet_utility(items)
        tab_max_nb_items, taille = get_max_number_item(items_sorted, cap)
        tab_poids_new = get_tab_poid_new(items_sorted, tab_max_nb_items)
        tab_gain_new = get_tab_gain_new(items_sorted, tab_max_nb_items)
        solutions = generer_solutions(nb_solutions, taille, tab_poids_new, cap)
        objects, solution, gain_tot, poids = geneticAlgorithm(items, cap, 20, solutions, nb_solutions, 2, 0.05, 0.8)
        end = time.time()
        st.write("le gain: ", gain_tot)
        st.write( "Temps d'éxécution : ", end - start)
        ################################################################################"""
        st.title("Fichier de taille entre 5000 et 10000 objets ")
        if(st.button("afficher graphe")):
            aa = ["Datasets\\Facile\\Moyenne\\cap591952_5000_facile.csv",
                  # "Datasets\\Facile\\Grande\\cap7547243_10000_facile.csv",
                  "Datasets\\Moyenne\\Moyenne\\cap3897377_5000_moy.csv",
                  "Datasets\\Moyenne\\Grande\\cap52926330_10000_moy.csv",
                  "Datasets\\Difficile\\Moyenne\\cap1596642_5000_diff.csv",
                  "Datasets\\Difficile\\Grande\\cap1596642_10000_diff.csv"]

            tmpp = []
            instancess = []
            for g in range(len(aa)):
                st.write("ggg", aa[g])
                cap, nb, items = readData6(aa[g])

                # wt = items[0].tolist()
                # val = items[1].tolist()

                start = time.time()
                nb_solutions = 100
                items_sorted = trier_objet_utility(items)
                tab_max_nb_items, taille = get_max_number_item(items_sorted, cap)
                tab_poids_new = get_tab_poid_new(items_sorted, tab_max_nb_items)
                tab_gain_new = get_tab_gain_new(items_sorted, tab_max_nb_items)
                solutions = generer_solutions(nb_solutions, taille, tab_poids_new, cap)
                objects, solution, gain_tot, poids = geneticAlgorithm(items, cap, 20, solutions, nb_solutions, 2, 0.05, 0.8)
                end = time.time()
                n = len(items[1])
                instancess.append(n)
                tmpp.append(end - start)

            fig = plt.figure(figsize=(10, 8), dpi=80)
            plt.plot(instancess, tmpp, color="forestgreen", label="WOH")
            #plt.px.bar(mean_counts_by_hour, x='hour', y='count', color='season', height=400)
            plt.xlabel("taille des instances (objets)")
            plt.ylabel("temps d'exécution (s)")
            plt.legend()
            st.pyplot(fig)

    elif page4=="hyper-heuristique AG de AG":
        st.title("Hyper-heuristique AG de AG")
        items = [[23, 92], [31, 57], [29, 49], [44, 68], [53, 60], [38, 43], [63, 67], [85, 84], [89, 87], [82, 72]]
        capacity = 165
        n = len(items)
        hyperGa = Hyper_GA(capacity, n, items)
        current_iteration, num_no_change,fitness = hyperGa.run()
        st.write("nb iter : ")
        st.write(current_iteration)
        st.write("num_no_change : ")
        st.write(num_no_change)
        st.write("fitness",fitness)

@st.cache(allow_output_mutation=True)
def get_data():
    return []
@st.cache
def load_data():
    # df = data.cars()
    return 0

def file_selector(folder_path='.'):
    filenames = glob.glob("D:\\optim_project\\*.csv")
    selected_filename = st.selectbox('Select a file', filenames)
    return selected_filename

def file_selector2(folder_path='.'):
    filenames = glob.glob("D:\\optim_project\\Datasets\\*\\*\\*.csv")

    selected_filename = st.selectbox('Select a file', filenames)
    return selected_filename

def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_circle(size=60).encode(
        x=x_axis,
        y=y_axis,
        color='Origin',
        tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    ).interactive()

    st.write(graph)

if __name__ == "__main__":
    main()