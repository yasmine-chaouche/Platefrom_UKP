from logging import NullHandler
from altair.vegalite.v4.schema.channels import Key
from AGG import read_benchmark_instance
from AGG import ukp_binarize
from AGG import create_ordered_index
from AGG import generate_random_population
from AGG import get_fitness_of
from AGG import cross_over
from AGG import mutate
from AGG import repair
from AGG import ukp
from AGG import ukp_ga
from AGG import exec_instance
from AGG import get_cost_weight_knapsack
from AGG import init_solution
from random import randrange
from timeit import default_timer as timer
import streamlit as st
import altair as alt
import os
import glob
import pandas as pd
import time
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import csv
#import SessionState


def main():

    page1 = st.sidebar.selectbox("choisir une méthode exacte", ["","Programmation dynamique","Branch & Bound"])
    page2 = st.sidebar.selectbox("choisir une heuristique",["","Order density","...." ])
    page3 = st.sidebar.selectbox("choisir une méthaheuristique", ["","Algorithme Génétique", "Récuit Simulé"])
    pages = ["page1", "page2", "page3"]
    title = ""
    page = st.sidebar.radio("Navigate", options=pages)
    if page1 == "Branch & Bound":
        st.title("Branch & Bound")


    elif page1 == "Programmation dynamique":
        title = st.title("Programmation Dynamique DP")

        filename = file_selector()
        data = pd.read_csv(filename)
        st.write(data)
        wt = data['volumes'].tolist()
        val = data['gains'].tolist()
        W = 129
        n = len(val)
        
    if page2== "Order density":
       
        title = st.title("Heuristique : Ordered density")
    elif page2==".....":
        
        title = st.title("Heuristique : Ordered density total")
    if page3=="Algorithme Génétique":
        title = st.title("Métaheuristique : Algorithme Génétique")
        html_temp = """
            <div>
            <h1 style="text-align:center;">Source de Données</h1>
            </div><br>"""
        st.markdown(html_temp,unsafe_allow_html=True)

        st.subheader("Entrez les données manuellement")
        
        #generations , prcnt_mutation  = st.beta_columns([2,2])
        #generations = generations.number_input("Nombre de générations")
        #prcnt_mutation = prcnt_mutation.number_input("Poucentage de mutation")

        i = bytes(0)
        with st.beta_container():
            weight , gain  = st.beta_columns([2,2])
            weight = weight.number_input("Poid", key=i)
            gain = gain.number_input("Gain", key=i+bytes(1))
            i = i + bytes(2)
        
        if st.button("Add Row") :
            weight , gain  = st.beta_columns([2,2])
            weight = weight.number_input("Poid",key=i)
            gain = gain.number_input("Gain",key=i+bytes(1))
            i = i + bytes(2)
        
        st.subheader("Entrez les données à partir de fichier csv")
        filename = file_selector()
        data = pd.read_csv(filename) 
        
        wt =  data['volumes'].tolist()
        val = data['gains'].tolist()

        #data = {
          #"\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t":["" for _ in range(len(wt))],
          #"volumes": wt,
          #"\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t":["" for _ in range(len(val))],
          #"gains": val,
        #}
     
        
        
        if st.button("Add row"):
         data =[]
         data.append({"volumes": weight, "gains": gain})

        df = pd.DataFrame(data)
        st.write(df)


        with st.form("Séléction des paramètres"):
            html_temp = """
            <div>
            <h1 style="text-align:center;">Séléction des parametres</h1>
            </div><br>"""
            st.markdown(html_temp,unsafe_allow_html=True)
            st.subheader("Le nombre de générations")
            generations = st.number_input("",min_value=None,help="Entrez le nombre de générations")
            st.subheader("Le pourcentage de mutation")
            mutation_percentage = st.number_input("",help="Entrez le pourcentage de mutation")

            # Every form must have a submit button.
            submitted = st.form_submit_button("Exécuter")
            if submitted:
                st.write("En execution",unsafe_allow_html=True)
        

        tempss=[] #temps d'éxec
        inst=[] #les capacités des instances

        x=0
        for x in range(len(a)):

           instance = readData(a[x])
           print("capacite du sac :"+str(instance[0]))
           capp=instance[0]
           inst.append(capp)
           listpoids=[]
           listprofits=[]
           dat=instance[2] #les objets sont stockés ici
           j=0
           for j in range(len(dat)):
              listpoids.append(dat[j].weight)
              listprofits.append(dat[j].value)


           instances= ukp(capp,listpoids,listprofits)
           exec_instance("ukp_ga.100.0.2", instances, 100, 0.2)

        
    elif page3=="Récuit Simulé":
        title = st.title("Métaheuristique : Récuit Simulé")

        
        html_title= """
            <div>
            <h1 style="text-align:center;">Source de Données</h1>
            </div><br>"""
        
        with st.beta_container():
            html_temp = """
            <div>
            <h1 style="text-align:center;">Source de Données</h1>
            </div><br>"""
            st.markdown(html_temp,unsafe_allow_html=True)

            st.subheader("Enterer les données manuellement")
            
            capacity , add_row  = st.beta_columns([2,2])
            capacity = capacity.number_input("Capacité")
            
            i = 0
            with st.beta_container():
                weight , gain  = st.beta_columns([2,2])
                weight = weight.number_input("Poid",key=i)
                gain = gain.number_input("Gain",key=i+1)
                i = i + 2
            
                
            if st.button("Add Row") :
                weight , gain  = st.beta_columns([2,2])
                weight = weight.number_input("Poid",key=i)
                gain = gain.number_input("Gain",key=i+1)
                i = i + 2

            st.subheader("Enterer les données à partir de fichier csv")
            filename = file_selector()
            data = pd.read_csv(filename)
        
            wt = data['volumes'].tolist()
            val = data['gains'].tolist()
            
            data = {
                "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t":["" for _ in range(len(wt))],
                "voulumes": wt,
                "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t":["" for _ in range(len(wt))],
                "gains": val,
            }
            df = pd.DataFrame( data)
            st.dataframe(df)

            W = 129
            n = len(val)


            # Every form must have a submit button.
            #submitted = st.form_submit_button("Valider")
            #if submitted:
            #    st.write("submitted")
        
        with st.form("Séléction des parametres"):
            html_temp = """
            <div>
            <h1 style="text-align:center;">Séléction des parametres</h1>
            </div><br>"""
            st.markdown(html_temp,unsafe_allow_html=True)
            st.subheader("Génération de la solution initial")
            sol_init_str  = st.radio('', ['Méthode aléatoire', 'Methode constructive'], 0)
            if sol_init_str == "Aléatoire":
                st.write("generation de solution aléatoirement")
            else:
                st.write("Méthode constructive")
            st.subheader("Température intiale")
            temp_init_str = st.number_input("",min_value=None,help="Enterez une valeur de température initial, sinon laissez le champs vide pour la calculer automatiquement")
            st.subheader("Température Finale")
            temp_final = st.number_input("",help="L'algorithme s'arrete lorsqu'on atteint cette température. En laissant ce champs vide, l'algorithme s'arrete à une température  de 0.5")
            st.subheader("Nombre d'itérations par paliers de température")
            nb_iter = st.number_input("",min_value=0,help="Nombre d'itérations qu'on reste dans une température avant la changer")
            st.subheader("Facteur alpha")
            alpha = st.number_input("",help="Facteur de diminution de la temperature. Si non spécifié, un nombre aléatoire entre 0.7 et 0.9 sera choisi")

            # Every form must have a submit button.
            submitted = st.form_submit_button("Exécuter")
            if submitted:
                st.write("En execution",unsafe_allow_html=True)
        with st.beta_expander("See explanation"):
            st.write("""The chart above shows some numbers I picked for you.I rolled actual dice for these, so they're *guaranteed* tobe random.""")
            st.image("https://static.streamlit.io/examples/dice.jpg")












def readData(file_name):
    file_name_arr_str = file_name.split("/",3)
    type_instance = file_name_arr_str[1]
    st.write("Instance de type : "+type_instance)
    size_type = file_name_arr_str[2]

    title = file_name_arr_str[3]
    title_data = title.split("_")
    capacity = int(title_data[0].split("cap",1)[1])
    instanceLength = int(title_data[1])
    st.write("Taille "+ size_type+" : "+str(instanceLength))

    data = []
    with open(file_name) as datasetFile :
       csv_reader = csv.reader(datasetFile,delimiter=",")
       line_count = 0
       for row in csv_reader:
          # columns name : 
          # volumes,gains
          if(line_count != 0 ):
             data.append(Item(int(row[0]),int(row[1])))
          ine_count = line_count + 1
    return capacity , instanceLength , data
a= ["Datasets/Facile/Moyenne/cap591952_5000_facile.csv"]

def file_selector(folder_path='.'):
    filenames = glob.glob("C:\\Users\\ASUS\\Downloads\\Compressed\\tp_optim_project\\tp_optim_project\\Datasets\\Facile\\Moyenne\\*.csv")
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_circle(size=60).encode(
        x=x_axis,
        y=y_axis,
        color='Origin',
        tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    ).interactive()

    st.write(graph)


def exec_instance(*k):
    type = k[0]
    ukp_o = k[1]
    start = timer()
    # (type, ukp_obj, generations, mutation)
    if "ukp_ga" in type:
        ukp_o_bin = ukp_binarize(ukp_o)
        selected = ukp_ga(ukp_o, ukp_o_bin, k[2], k[3])
        solution = ukp_debinarize_solution(ukp_o_bin, selected)

    else:

        return
    time = str(timedelta(seconds=timer() - start))

    global global_fhandle

    if global_fhandle != 0:
     global_fhandle.write(type + ',' + time + ',' + str(solution.total) + ',' + str(solution.totalw) + '\n')
    #print (type + ',' + time + ',' + str(solution.total) + ',' + str(solution.totalw))
    st.write("le temps d'execution : " + time )
    st.write("le total de profit : "+str(solution.total))
    st.write("le total de poids : "+str(solution.totalw))
   # for i in range(0,len(solution.taken)):
        #print("L'objet" + str(solution.taken[i])+ "est pris : "  + str(solution.ttimes[i])+ "fois")
    
    return {"sol":solution, "time":time}




if __name__ == "__main__":
 main()

def ukp_ga(ukp_obj, bin_obj, generations, mutation_percentage):

    OrdIndex = create_ordered_index(bin_obj)
    pop1 = generate_random_population (bin_obj)
    pop2 = generate_random_population (bin_obj)

    generation = 0
    while generation < generations :

        fpop1 = get_fitness_of(bin_obj, pop1)
        fpop2 = get_fitness_of(bin_obj, pop2)

        best = pop1 if fpop1 > fpop2 else pop2 # Meilleur individu pour cette itération

        child = cross_over (pop1, pop2)
        child = mutate (child, mutation_percentage)

        repair (bin_obj, child, OrdIndex)

        # Remplacer le pire d'individu par child (fils)
        if fpop1 > fpop2 :
            pop2 = child
        else :
            pop1 = child
        fsc = get_fitness_of(bin_obj, child)

        # Remplacer le meilleur individu par child si f(child) > f(best)
        if fpop1 > fpop2 and fsc > fpop1:
            pop1 = child
        elif fpop2 > fpop1 and fsc > fpop2:
            pop2 = child

        generation += 1

    return best

''' This code is to be added in the statistics 

progress_bar = st.progress(0)
status_text = st.empty()
chart = st.line_chart(np.random.randn(10, 2))

for i in range(100):
    # Update progress bar.
    progress_bar.progress(i + 1)

    new_rows = np.random.randn(10, 2)

    # Update status text.
    status_text.text('The latest random number is: %s' % new_rows[-1, 1])

    # Append data to the chart.
    chart.add_rows(new_rows)

    # Pretend we're doing some computation that takes time.
    time.sleep(0.1)

st.balloons()
'''
