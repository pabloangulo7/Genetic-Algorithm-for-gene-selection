import os
import sys
import pandas as pd
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms


def compute_HammingDistance(X):
    return (X[:, None, :] != X).sum(2)

dm_list = list()

for gene_id,gene in gene_dict.items():
    snp = list()
    for sample in sample_list:
    	snp.append(gene_dict[sample]['genotype_vector'])
    	dm = compute_HammingDistance(np.array(snp))
    	dm_list.append([dm,gene_id])

dm_dict = dict()
index = 0
for i in dm_list:
    dm_dict[index] = i
    index += 1

cluster = [1,3,4,2,4,1,2,3,1,2,1,3,2,4,
           1,2,3,2,3,3,1,3,3,3,3,2,4,3,
           2,1,1,2,2,3,1,2,3,4,4,2,3,3,
           3,3,1,4,1,4,3,2,4,2,3]

IND_SIZE = len(dm_list)
pop_size = 1500
mutrate = 0.05
cxpb = 0.7
mutpb = 0.07
ngen = 1000
hof = tools.HallOfFame(10)
k_sel = 1500
tournsize = int(0.20*pop_size)

matrix_p = np.mean(np.array([value[0] for key,value in dm_dict.items()]),axis=0)
silhouette_p = metrics.silhouette_score(matrix_p,cluster,metric="precomputed")

def evaluate(individual):
    matrix = np.mean(np.array([value[0] for key,value in dm_dict.items() if individual[key] == 1]),axis=0)
    silhouette = metrics.silhouette_score(matrix,cluster,metric="precomputed")
    return silhouette,


creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int",random.randint,a=0,b=1)
toolbox.register("individual", tools.initRepeat,creator.Individual,toolbox.attr_int,n=IND_SIZE)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)
toolbox.register("mate",tools.cxTwoPoint)
toolbox.register("mutate",tools.mutUniformInt,low=0,up=1,indpb=mutrate)
toolbox.register("select",tools.selTournament,tournsize=tournsize,fit_attr='fitness')
toolbox.register("evaluate",evaluate)

pop = toolbox.population(pop_size)


algorithms.eaSimple(population=pop,toolbox=toolbox,
                    cxpb=cxpb,mutpb=mutpb,ngen=ngen,
                    halloffame=hof,verbose=True)

hof.keys







