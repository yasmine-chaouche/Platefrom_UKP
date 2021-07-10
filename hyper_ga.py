def sort_items_by_profit(items):
    items.sort(key= lambda item : item[1]/item[0],reverse=True)


from RSS import simulatedAnnealing
from AGG import geneticAlgorithm, generer_solutions, eval_solution, get_tab_gain_new, get_tab_poid_new, get_max_number_item, trier_objet_utility
from Utils import gen_random_sol
from RSS import gen_random_sol, simulatedAnnealing
import random
import time


class LLMH:
    @staticmethod
    def run(capacity, n, items, solinit):
        return None


class AG1(LLMH):
    @staticmethod
    def run(capacity, n, items, solinit):
        nb_tour = 20
        nb_solutions = 50
        nb_per_group = 2
        proba_mutation = 0.05
        pc = 0.8
        # objects,solution,binarized_solution,gain_tot,poids  =
        return geneticAlgorithm(items, capacity, nb_tour, solinit, nb_solutions, nb_per_group, proba_mutation, pc)


class AG2(LLMH):
    @staticmethod
    def run(capacity, n, items, solinit):
        nb_tour = 20
        nb_solutions = 50
        nb_per_group = 2
        proba_mutation = 0.1
        pc = 0.6
        # objects,solution,binarized_solution,gain_tot,poids  =
        return geneticAlgorithm(items, capacity, nb_tour, solinit, nb_solutions, nb_per_group, proba_mutation, pc)
        # return solution


class AG3(LLMH):
    @staticmethod
    def run(capacity, n, items, solinit):
        nb_tour = 20
        nb_solutions = 50
        nb_per_group = 2
        proba_mutation = 0.15
        pc = 0.8
        # objects,solution,binarized_solution,gain_tot,poids  =
        return geneticAlgorithm(items, capacity, nb_tour, solinit, nb_solutions, nb_per_group, proba_mutation, pc)


class AG4(LLMH):
    @staticmethod
    def run(capacity, n, items, solinit):
        nb_tour = 20
        nb_solutions = 50
        nb_per_group = 2
        proba_mutation = 0.2
        pc = 0.6
        # objects,solution,binarized_solution,gain_tot,poids  =
        return geneticAlgorithm(items, capacity, nb_tour, solinit, nb_solutions, nb_per_group, proba_mutation, pc)




def updateSolutions(solutions, newSol, newEval, tab_gain_new):
    loop = True
    i = 0
    # print("new solution")
    # print(newSol)
    # print("***********************")
    while loop and i < len(solutions):
        # print("solution")
        # print(len(solutions[i]))
        # print("***********************")
        # print("gain")
        # print(len(tab_gain_new))
        # print("***********************")
        if (newEval > eval_solution(solutions[i], tab_gain_new)):
            solutions[i] = newSol
            loop = False
        i = i + 1
    return solutions


# from ipynb.fs.full.low_level import AG1,AG2,RS1,RS2
# from ipynb.fs.full.Utils import gen_random_sol
import random


class Hyper_GA:
    POPULATION_SIZE = 20
    MAX_GENERATIONS = 50
    MAX_NO_EVOLUTION = 40
    TOURNAMENT_SIZE = 20
    MUTATION_RATE = 0.05
    CROSSOVER_RATE = 0.8
    MAX_NO_CHANGE = 5

    def __init__(self, capacity, n, items):
        """
        Creates an instance that can run the genetic algorithm.
        :param capacity: The capacity of a knapsack
        :param items: The items that should be used to fill the knapsack.
        """
        self.items = items
        self.capacity = capacity
        self.instance_length = n
        self.best_solution = None
        self.population = [Chromosome(capacity) for _ in range(self.POPULATION_SIZE)]
        self.update_population(self.population)

    def run(self):
        """
        Runs the genetic algorithm and returns the results at the end of the process.
        :return: (num_iterations, num_no_changes)
        """
        current_iteration = 0
        num_no_change = 0
        while num_no_change < self.MAX_NO_CHANGE and current_iteration < self.MAX_GENERATIONS:
            new_generation = []
            while len(new_generation) < self.POPULATION_SIZE:
                # Select parents
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                # Apply genetic operators
                child1, child2 = self.crossover(parent1, parent2)
                child1, child2 = self.mutate(child1), self.mutate(child2)
                # Update the fitness values of the offspring to determine whether they should be added
                self.update_population([child1, child2])
                sorted_list = sorted([parent1, parent2, child1, child2], key=lambda x: x.fitness, reverse=True)
                # Add to new generation the two best chromosomes of the combined parents and offspring
                new_generation.append(sorted_list[0])
                new_generation.append(sorted_list[1])
            self.population = new_generation
            prev_best = self.best_solution
            # Evaluate fitness values
            self.best_solution = self.update_population(self.population)
            # Check if any improvement has happened.
            if not prev_best or prev_best.fitness == self.best_solution.fitness:
                num_no_change += 1
            else:
                num_no_change = 0
            current_iteration += 1
        return current_iteration, num_no_change,self.best_solution.fitness

    def mutate(self, chromosome):
        """
        Attempts to mutate the chromosome by replacing a random metaheuristic in the chromosome by a generated pattern.
        :param chromosome: The chromosome to mutate.
        :return: The mutated chromosome.
        """
        # pattern = list(chromosome.pattern)
        pattern = [chromosome.pattern[i:i + 2] for i in range(0, len(chromosome.pattern), 2)]
        # print("old pattern : ")
        # print(pattern)

        if random.random() < self.MUTATION_RATE:
            mutation_point = random.randrange(len(pattern))
            # print("mutation point : ")
            # print(mutation_point)
            newPattern = Chromosome.generate_pattern()
            new_mhs = [newPattern[i:i + 2] for i in range(0, len(newPattern), 2)]
            print("sommes des pattern : " + str(len(new_mhs) + len(pattern)))
            if (len(new_mhs) + len(pattern) >= chromosome.MAX_GENES):
                pattern[mutation_point] = new_mhs[0]
            else:
                pattern = pattern[:mutation_point] + new_mhs + pattern[mutation_point + 1:]
            # print("new pattern : ")
            # print(pattern)
            # print("******************************************")
            # pattern[mutation_point] = Chromosome.generate_pattern()
        return Chromosome(chromosome.capacity, "".join(pattern))

    def crossover(self, parent1, parent2):
        """
        Attempt to perform crossover between two chromosomes.
        :param parent1: The first parent.
        :param parent2: The second parent.
        :return: The two individuals after crossover has been performed.
        """
        pattern1, pattern2 = parent1.pattern, parent2.pattern
        pattern1_list = [parent1.pattern[i:i + 2] for i in range(0, len(parent1.pattern), 2)]
        pattern2_list = [parent2.pattern[i:i + 2] for i in range(0, len(parent2.pattern), 2)]
        if random.random() < self.CROSSOVER_RATE:
            point1, point2 = random.randrange(len(pattern1_list)), random.randrange(len(pattern2_list))
            substr1, substr2 = pattern1[point1:], pattern2[point2:]
            pattern1, pattern2 = "".join((pattern1[:point1], substr2)), "".join((pattern2[:point2], substr1))
            if len(pattern1) > 8:
                pattern1 = pattern1[:8]
            if len(pattern2) > 8:
                pattern2 = pattern2[:8]
        return Chromosome(parent1.capacity, pattern1), Chromosome(parent2.capacity, pattern2)

    def update_population(self, individuals):
        """
        Update the fitness values of all the chromosomes in the population.
        """
        for individual in individuals:
            solution, fitness = individual.generate_solution(self.capacity, self.instance_length, self.items)
            # no need for this instruction
            # ndividual.num_bins = len(solution)
            individual.fitness = fitness
            # individual.fitness = sum([sol[1] for sol in solution ] )#/len(solution)
        return max(self.population, key=lambda x: x.fitness)

    def select_parent(self):
        """
        Selects a parent from the current population by applying tournament selection.
        :return: The selected parent.
        """
        candidate = random.choice(self.population)
        for _ in range(self.TOURNAMENT_SIZE - 1):
            opponent = random.choice(self.population)
            if opponent.fitness > candidate.fitness:
                candidate = opponent
        return candidate


class Chromosome:
    MAX_GENES = 4
    metaheuristic_genes = {
        "00": AG1,
        "01": AG2,
        "10": AG3,
        "11": AG4,
    }

    def __init__(self, capacity, pattern=None):
        self.capacity = capacity
        self.fitness = 0
        self.pattern = pattern or self.generate_pattern()

    @staticmethod
    def generate_pattern():
        """
        Generates a random pattern.
        :return: The generated pattern string.
        """
        # print("chromosome keys : ")
        # print(list(Chromosome.metaheuristic_genes.keys()))
        # print("**********************************")
        return "".join(
            [random.choice(list(Chromosome.metaheuristic_genes.keys())) for _ in
             range(random.randrange(1, Chromosome.MAX_GENES))])

    def generate_solution(self, capacity, n, items):
        """
        Generates a candidate solution based on the pattern given.
        :param items: The items that need to be used when generating a solution.
        :return: items that form the solution.
        """
        items_sorted = trier_objet_utility(items)

        tab_max_nb_items, taille = get_max_number_item(items_sorted, capacity)

        tab_poids_new = get_tab_poid_new(items_sorted, tab_max_nb_items)

        tab_gain_new = get_tab_gain_new(items_sorted, tab_max_nb_items)
        # print("gains : ")
        # print(tab_gain_new)
        # print("************************************")
        nb_solutions = 50
        solutions = generer_solutions(nb_solutions, taille, tab_poids_new, capacity)

        bestSol = solutions[0]
        bestEval = eval_solution(bestSol, tab_gain_new)

        pattern_length = len(self.pattern)
        # print("pattern : "+str(self.pattern))
        nb = 2
        mhs = [self.pattern[i:i + nb] for i in range(0, len(self.pattern), nb)]
        # print("les meta heuristiques : ")
        # print(mhs)
        for mh in mhs:
            if not mh in self.metaheuristic_genes.keys():
                mhs.remove(mh)
        for mh in mhs:
            # print("current metaheuristic : ")
            # print(mh)
            # print("*******************************")

            # objects,solution,best_solution,gain_tot,poids

            result = self.metaheuristic_genes[mh].run(capacity, n, items, solutions)
            solutions = updateSolutions(solutions, result[2], result[3], tab_gain_new)
            bestSol = result[1]
            bestEval = result[3]
        # for idx, item in enumerate(items):
        #    mh = self.pattern[idx % pattern_length]
        #    solution = self.metaheuristic_genes[mh].run(capacity,n, items,solution)
        # print("Final solution : ")
        # print(bestEval)
        # print("***************************")
        return bestSol, bestEval
