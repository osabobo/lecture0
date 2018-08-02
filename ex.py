# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:30:00 2018

@author: Tolu
"""

# --- IMPORT LIBRARIES --------------------------------------------------------

import pandas_datareader.data as dr # Import pandas asset datareader
import datetime                     # Import datetime module 
import matplotlib.pyplot as plt     # Import matplotlib ploting library  
import numpy as np                  # Import numpy
import random                       # Import random 
import scipy as sp                  # Import scipy
import csv                          # Import csv
from itertools import zip_longest   # Import zip_longest
import pandas as pd
# --- GOBAL PARAMETERS --------------------------------------------------------

ALLELE_SIZE = 5        # Number of allele + 1 
POPULATION_SIZE = 10    # Population size
ELITE_CHROMOSOMES = 1  # Elite chromosome not subjected to evolving 
SELECTION_SIZE = 4     # Selection size randomly from population
MUTATION_RATE = 0.25   # Probability of random chromosome mutation
TARGET_FITNESS = 0.9   # Target fitness (change as need to get convergence)
ALPHA = 0.1            # Constant in the fitness test

# Random number generator list (change as needed)
R = [175,280,320,340,360,385] 

# Download daily TSLA stock prices for 1 years from Google Finance

symbol = "TSLA"                       # Ticker symbol
start  = datetime.datetime(2016,6,13) # Start date (adjust date for a year)
end    = datetime.date.today()        # Today's date

#data   = dr.DataReader(symbol, "google", start, end) # Retrieve data
data=pd.read_csv("C:/Users/Tolu/Desktop/TSLA.csv")
# Assest closing price and standard deviation
 
PRICE  = data["Close"].values.tolist()
SIGMA0 = np.std(PRICE) # Standard deviation of the entire timeseries

# --- CLASSES -----------------------------------------------------------------

class Chromosome:
    
    """
    Candidate solution.
    """
    
    def __init__(self):
        self._genes = []
        self._fitness = 0        
        self._genes.append(random.randint(R[0],R[1]))
        self._genes.append(random.randint(R[1],R[2]))
        self._genes.append(random.randint(R[2],R[3]))
        self._genes.append(random.randint(R[3],R[4]))
        self._genes.append(random.randint(R[4],R[5]))

    def get_genes(self):
        return self._genes
    
    def get_fitness(self):
        L = []
        D = []
        for i in range(len(PRICE)-2):
            if self._genes[0] <= PRICE[i]:
                if self._genes[1] <= PRICE[i+1] or PRICE[i+1] <= self._genes[2]:
                    if self._genes[3] <= PRICE[i+2] or PRICE[i+2] <= self._genes[4]:
                        L.append(PRICE[i+1])
                        D.append(i)    
        sigma = np.std(L)
        nc = len(L)
        self._fitness = -np.log2(sigma/SIGMA0)-ALPHA/nc        
        return self._fitness

    def __str__(self):
        return self._genes.__str__()
          
class Population:
    
    """
    Pupulation of candidate solutions.
    """
    
    def __init__(self, size):
        self._chromosomes = []
        i = 0
        while i < size:
            self._chromosomes.append(Chromosome())
            i += 1
    
    def get_chromosomes(self):
        return self._chromosomes

class GeneticAlgorithm:
    
    """
    The genectic algorithm logic for evolving the population via crossover and
    mutation.
    """
    
    @staticmethod
    def evolve(pop):
        return GeneticAlgorithm._mutate_population(GeneticAlgorithm._crossover_population(pop))
    
    @staticmethod
    def _crossover_population(pop):
        crossover_pop = Population(0)
        for i in range(ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
        i = ELITE_CHROMOSOMES
        while i < POPULATION_SIZE:
            chromosome1 = GeneticAlgorithm._select_population(pop).get_chromosomes()[0]
            chromosome2 = GeneticAlgorithm._select_population(pop).get_chromosomes()[0]
            crossover_pop.get_chromosomes().append(GeneticAlgorithm._crossover_chromosomes(chromosome1, chromosome2))
            i += 1
        return crossover_pop
    
    @staticmethod
    def _mutate_population(pop):
        for i in range(ELITE_CHROMOSOMES, POPULATION_SIZE):
            GeneticAlgorithm._mutate_chromosomes(pop.get_chromosomes()[i])
        return pop
    
    @staticmethod
    def _crossover_chromosomes(chromosome1, chromosome2):
        crossover_chrom = Chromosome()
        for i in range(ALLELE_SIZE):
            if random.random() < MUTATION_RATE:
                if random.random() < 0.5:
                    crossover_chrom.get_genes()[i] = chromosome1.get_genes()[i]
                else:
                    crossover_chrom.get_genes()[i] = chromosome2.get_genes()[i]
        return crossover_chrom
    
    @staticmethod
    def _mutate_chromosomes(chromosome):
        if random.random() < MUTATION_RATE:
            chromosome.get_genes()[0] = random.randint(R[0],R[1])
        elif random.random() < MUTATION_RATE:
            chromosome.get_genes()[1] = random.randint(R[1],R[2])
        elif random.random() < MUTATION_RATE:
            chromosome.get_genes()[2] = random.randint(R[2],R[3])
        elif random.random() < MUTATION_RATE:
            chromosome.get_genes()[3] = random.randint(R[3],R[4])
        elif random.random() < MUTATION_RATE:
            chromosome.get_genes()[4] = random.randint(R[4],R[5])
        
    @staticmethod
    def _select_population(pop):
        select_pop = Population(0)
        i = 0
        while i < SELECTION_SIZE:
            select_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0,POPULATION_SIZE)])
            i += 1
        select_pop.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        return select_pop
def print_population(pop, gen_number):
    
    """
    Print evolving generation results.
    """
    
    print("\n-----------------------------------------------------------")
    print("Generation #", gen_number, ": Fittest chromosome fitness: %3.2f" % pop.get_chromosomes()[0].get_fitness())
    print("-----------------------------------------------------------")
    i=0
    for x in pop.get_chromosomes():
        print("Chromosome #", i, " :", x, "| Fitness: %3.2f" % x.get_fitness())
        i += 1
# --- MAIN --------------------------------------------------------------------

# Generation 0
        
population = Population(POPULATION_SIZE)
population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
print_population(population, 0)
# Price list that met the criteria with the heigest fitness
genes = []
for i in range(5):
    genes.append(population.get_chromosomes()[0].get_genes()[i])

L = [] # List of price that met cretria with the target fitness
D = [] # List of numbered days that met the criteria (starting form day 0)
for i in range(len(PRICE)-2):
    if genes[0] <= PRICE[i]:
        if genes[1] <= PRICE[i+1] or PRICE[i+1] <= genes[2]:
            if genes[3] <= PRICE[i+2] or PRICE[i+2] <= genes[4]:
                L.append(PRICE[i+1])
                D.append(i)
# 1st order polynimial fit or linear regression of data meeting the fitness
# criteria
            
p1 = sp.polyfit(D,L,1)
LF = sp.polyval(p1,D) # Fitted data from D

# 1st order polynimial fit or linear regression of forecast (2 years) from the
# last day of actual data

E  = list(range(272,776))
LE = sp.polyval(p1,E) # Fitted data to E

# Plotting of the daily close results

plt.plot(PRICE)
plt.title("TSLA")
plt.xlabel("Days")
plt.ylabel("Price")
plt.grid(True)
plt.show()
# Plotting of data meeting fitness criteria, fitted, and forcast results

an = "Price = " + "%0.2f" % (p1[0]) + " * Days + " + "%0.2f" % (p1[1])
plt.plot(D,L,'ro',label="Fitness Met Data")
plt.title("TSLA")
plt.xlabel("Days")
plt.ylabel("Price")
plt.grid(True)
plt.plot(D,LF,"b-",label="Fitted Curve")
plt.plot(E,LE,"b-",label="Forecast Curve",linestyle='--')
plt.legend()
plt.text(510, 510, an, color='green')
plt.show()
# Exporting to CSV format of the results

with open('GAOut.csv', 'w') as out:
    writer = csv.writer(out)
    writer.writerow(["Day", "Fitness Met Data", "Fitted Curve","Forecast Curve"])
    for row in zip_longest(D+E,L,LF,LF.tolist()+LE.tolist()):
        writer.writerow(row)  
         
# --- END ---------------------------------------------------------------
