import numpy as np
from numpy import random
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import pearsonr, ks_2samp
import math
import matplotlib.pyplot as plt
from pandas import *
import itertools
from nltk import FreqDist
from nltk.corpus import brown
from random import sample
try:
    nltk.download('universal_tagset')
except Exception:
    pass
from sqlalchemy import create_engine, text
from sqlalchemy.schema import CreateSchema, DropSchema
import graphviz
import random
import warnings
warnings.filterwarnings("ignore")
df = pd.DataFrame()
tables_number = 50
repeated_coeff = 0.1
round_coeff = 1
schema_number = 5
hypothesis_coeff = 1.00
columns_min = 2
columns_max = 5
array_amount = 1000
mixed_distibutions_amount = 3
low = 0.2

distibutions_type = "one"  # ["one", "mixed"]

db_name = 'postgres'
schemas_all = []

function_words = ['data', 'index', 'columns', 'copy', 'where', 'first']
frequency_list = FreqDist(i.lower() for i in brown.words())
data_c = sample([i for i, j in frequency_list.items() if j >= 1 and len(i) >= 1 and i.find('`')==-1 and i.find("'")==-1 and i not in function_words], int(tables_number * columns_max))
data_t = sample([i for i, j in frequency_list.items() if j >= 10 and len(i) >= 3 and i.find('`')==-1 and i.find("'")==-1 and i not in function_words and i not in data_c], tables_number)
data_s = sample([i for i, j in frequency_list.items() if j >= 100 and len(i) >= 5 and i.find('`')==-1 and i.find("'")==-1 and i not in function_words and i not in data_c and i not in data_t], schema_number)

all_dist = [getattr(stats, d) for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]
distsributions = [x.name for x in all_dist]
distributions_processed = []

for j in distsributions:
    try:
        getattr(scipy.stats, j).rvs(size=1)
        distributions_processed.append(j)
    except:
        pass
    
# distributions = distributions_processed
distributions = ['norm', 'uniform', 'logistic', 'halfnorm', 'expon']

d = random.choices(distributions, k = int(len(data_c) - (len(data_c) * repeated_coeff)))

class DistributionsMixed(scipy.stats.rv_continuous):
    def __init__(self, submodels, *args, weights = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.weights = [w / sum(weights) for w in weights]
        
    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x)  * weight
        return pdf        

    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.submodels), size=size, p = self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs

for count, j in enumerate(d):
    if distibutions_type == 'one':
        data_column = getattr(scipy.stats, j).rvs(loc=random.choice([h for h in range(-5, 5)]), scale=random.choice([h for h in range(1, 2)]), size=array_amount)
    elif distibutions_type == 'mixed':
        a = np.random.rand(mixed_distibutions_amount)
        a = (a/a.sum()*(1-low*mixed_distibutions_amount))
        weights = a+low
        distibutions_mixed = random.sample(distributions, mixed_distibutions_amount)
        for d in distibutions_mixed:
            mixture_model = DistributionsMixed([getattr(stats, j)(random.choice([h for h in range(-5, 5)]), random.choice([h for h in range(1, 2)])) for j in distibutions_mixed], weights = weights)
            array = mixture_model.rvs(array_amount)
            #plt.hist(array, bins = 50, density = True)
            #plt.show()
        data_column = array
    #print(count, f"{j}_{count}")
    df.loc[:, f"{j}_{count}"] = np.array(np.round(data_column, round_coeff)).tolist()
    #plt.hist(data_column, bins = 50, density = True, label = 'Distribution')
    #plt.show()
    
for i in range(int(len(data_c) * repeated_coeff)):
    random_column = sample(list(df.columns), 1)[0]
    df[df.shape[1]] = df[random_column].copy()

for col, col2 in zip(df.columns, data_c):
    df.rename({col: col2}, axis=1, inplace=True)
    
correlations = {}
columns = df.columns.tolist()

for col_a, col_b in itertools.combinations(columns, 2):
    correlations[col_a + '->' + col_b] = stats.ks_2samp(df.loc[:, col_a], df.loc[:, col_b], method='exact', alternative='two-sided')

result = DataFrame.from_dict(correlations, orient='index')
result.columns = ['statistics', 'p_value']
result = result[result.p_value >= hypothesis_coeff]

connectable = create_engine("postgresql+psycopg2://***:***@***:***/***")
for i in data_s:
    with connectable.connect() as connection:

        schemas = list(connection.execute(text(f"SELECT * FROM information_schema.schemata where catalog_name = '{db_name}' and schema_name not like '%pg_%' and schema_name != 'information_schema'")).fetchall())

        for i in schemas:
            for j in i:
                if j not in schemas_all:
                    schemas_all.append(j)

with connectable.connect() as connection:
    for s in schemas_all:
        try:
            connection.execute(DropSchema(s, cascade=True, if_exists=True))
            connection.commit()
        except AttributeError:
            pass
        
tables_per_schema = {}
        
for i in data_s:
    if i not in tables_per_schema:
        tables_per_schema[i] = []
    with connectable.connect() as connection:
        connection.execute(CreateSchema(i, if_not_exists=True))
        connection.commit()
for i in range(tables_number):
    with connectable.connect() as connection:
        schema_index = str(random.choice([i for i in data_s]))
        table_name = str(random.choice([i for i in data_t]))
        distribution_indexes = list(random.choices(data_c, k = random.randint(columns_min, columns_max)))
        try:
            for d in distribution_indexes:
                tables_per_schema[schema_index].append(d)
                data_c.remove(d)
            df[tables_per_schema[schema_index]].to_sql(table_name, connection, schema=schema_index, if_exists='fail', index=False)
            data_t.remove(table_name)
            connection.commit()
        except:
            pass
dot = graphviz.Digraph('BenchER - Benchmark for ER-models', comment='Author: Moskalev Dmitry', format='svg')  
tables_per_schema = {k: v for k, v in tables_per_schema.items() if len(v) != 0}

for i in result.index:
    col1 = str(i.split('->')[0])
    col2 = str(i.split('->')[1])
    try:
        for key, value in tables_per_schema.items():
            if col1 in value:
                schema1 = key
        if col2 in value:
                schema2 = key
    
        if result.loc[i].p_value >= hypothesis_coeff:
            #print(schema1 + '.' + col1 + ';' + schema2 + '.' + col2)  # GT
            dot.edge(schema1+ '.' + col1, schema2 + '.' + col2, label=f'{round(result.loc[i].p_value, 2)}')
    except NameError:
        pass

dot.view()
