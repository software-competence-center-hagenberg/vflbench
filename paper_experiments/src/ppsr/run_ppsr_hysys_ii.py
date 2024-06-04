# Import libraries
import pickle
import random
import operator
import jax
import numbers
import timeit
import signal
import functools
import time
import argparse
import numpy as np
import pandas as pd
import secretflow as sf
import jax.numpy as jnp
import sympy as sp
import spu as spu_lib
from deap import algorithms, base, creator, tools, gp
from functools import partial
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from ppsr.sympy2jax import sympy2jax

# Extract arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run", "-r", help="Run no.")
args = parser.parse_args()

run_no = args.run

# Set random seed
RANDOM_SEED = int(run_no)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(parties=["c_1", "c_2"], address="local")
c_1, c_2 = sf.PYU("c_1"), sf.PYU("c_2")

spu_config = sf.utils.testing.cluster_def(["c_1", "c_2"])
spu = sf.SPU(spu_config)

# spu_config["runtime_config"]["protocol"] = spu_lib.spu_pb2.CHEETAH
# spu_config['runtime_config']['field'] = spu_lib.spu_pb2.FM64
# spu_config['runtime_config']['fxp_fraction_bits'] = 18
#
# print(spu_config)

# Load data
dataset_name = "hysys_ii"
input_path = "/workspace/federated_data/{}.pkl".format(dataset_name)
with open(input_path, "rb") as f:
    fed_data = pickle.load(f)

cross_valid_data = fed_data["cross_valid_data"]
n_clients = fed_data["num_parties"]
task = fed_data["task"]
method_name = "ppsr_sigmoid"

Xs_train, y_train, Xs_test, y_test = cross_valid_data[0]


# Scale data
def scale_data(train_data, test_data):
    n_parties = len(train_data)

    train_data_scaled = []
    test_data_scaled = []
    for i in range(n_parties):
        scaler = StandardScaler()
        scaler.fit(train_data[i])
        train_data_scaled.append(scaler.transform(train_data[i]))
        test_data_scaled.append(scaler.transform(test_data[i]))

    return train_data_scaled, test_data_scaled


Xs_train, Xs_test = scale_data(Xs_train, Xs_test)

# Separate data
X_1_train = Xs_train[0]
X_1_test = Xs_test[0]
X_2_train = Xs_train[1]
X_2_test = Xs_test[1]

m_train = X_1_train.shape[0]  # No. training observations
m_test = X_1_test.shape[0]  # No. test observations
n_c_1 = X_1_train.shape[1]  # No. features that belong to c_1
n_c_2 = X_2_train.shape[1]  # No. features that belong to c_2
n_features = n_c_1 + n_c_2

# Send data to PYU devices
X_1_train_enc = sf.to(c_1, X_1_train)
X_1_test_enc = sf.to(c_1, X_1_test)

X_2_train_enc = sf.to(c_2, X_2_train)
X_2_test_enc = sf.to(c_2, X_2_test)
y_train_enc = sf.to(c_2, y_train)
y_test_enc = sf.to(c_2, y_test)

threshold = 0.5
threshold_enc = sf.to(c_2, threshold)

# Sympy converter
converter = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "sin": lambda x: sp.sin(x),
    "cos": lambda x: sp.cos(x)
}

# Sympy's symbols
for j in range(n_features):
    globals()["x_{}".format(j + 1)] = sp.symbols("x_{}".format(j + 1))

# Define primitive operators
pset = gp.PrimitiveSet("TRAIN", n_features)
pset.addPrimitive(np.add, 2, name="add")
pset.addPrimitive(np.subtract, 2, name="sub")
pset.addPrimitive(np.multiply, 2, name="mul")
pset.addPrimitive(np.sin, 1, name="sin")
pset.addPrimitive(np.cos, 1, name="cos")
pset.addEphemeralConstant("rand_float_{}".format(dataset_name), partial(random.uniform, -5, 5))
arguments_dict = {"ARG{}".format(i): "x_{}".format(i + 1) for i in range(n_features)}
pset.renameArguments(**arguments_dict)

# Define SR
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# Define the fitness function
def evalSymbReg(individual):
    # Convert String to Sympy
    # func = sp.expand(sp.sympify(str(individual), locals=converter))
    func = sp.sympify(str(individual), locals=converter)

    # Convert Sympy to Jax
    f, params = sympy2jax(func, [globals()["x_{}".format(j + 1)] for j in range(n_features)])

    # Define sigmoid function
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))

    def calculate_fitness(X_1_en, X_2_en, y_en, threshold_en):
        # Concatenate features
        X_enc = jnp.concatenate([X_1_en, X_2_en], axis=1)

        # Calculate predictions
        y_scores_pred_en = f(X_enc, params)
        y_pred_en = jnp.greater_equal(sigmoid(y_scores_pred_en), threshold_en)

        return jnp.mean(jnp.equal(jnp.ravel(y_en), jnp.ravel(y_pred_en)))

    fitness_enc = spu(calculate_fitness)(X_1_train_enc, X_2_train_enc, y_train_enc, threshold_enc)

    fitness = sf.reveal(fitness_enc)

    return fitness,


# Make predictions
def make_predictions(solution):
    # Convert String to Sympy
    # func = sp.expand(sp.sympify(solution, locals=converter))
    func = sp.sympify(solution, locals=converter)

    # Convert Sympy to Jax
    f, params = sympy2jax(func, [globals()["x_{}".format(j + 1)] for j in range(n_features)])

    # Define sigmoid function
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))

    def predict(X_1_en, X_2_en):
        # Concatenate features
        X_enc = jnp.concatenate([X_1_en, X_2_en], axis=1)

        # Calculate predictions
        y_scores_pred_en = sigmoid(f(X_enc, params))

        return y_scores_pred_en

    y_train_scores_pred_enc = spu(predict)(X_1_train_enc, X_2_train_enc)
    y_test_scores_pred_enc = spu(predict)(X_1_test_enc, X_2_test_enc)

    y_train_scores_pred = sf.reveal(y_train_scores_pred_enc)
    y_test_scores_pred = sf.reveal(y_test_scores_pred_enc)

    if isinstance(y_train_scores_pred, numbers.Number):
        y_train_scores_pred = [y_train_scores_pred] * m_train
    elif len(y_train_scores_pred) == 1:
        y_train_scores_pred = [y_train_scores_pred[0]] * m_train

    if isinstance(y_test_scores_pred, numbers.Number):
        y_test_scores_pred = [y_test_scores_pred] * m_test
    elif len(y_test_scores_pred) == 1:
        y_test_scores_pred = [y_test_scores_pred[0]] * m_test

    y_train_scores_pred = np.array(y_train_scores_pred)
    y_test_scores_pred = np.array(y_test_scores_pred)

    y_train_pred = y_train_scores_pred >= threshold
    y_test_pred = y_test_scores_pred >= threshold

    return y_train_pred, y_test_pred


toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

limit_height = 15
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

limit_length = 120
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_length))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_length))

# Test
acc_list = []
f1_list = []

# Training
pop = toolbox.population(n=1000)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("Avg", np.mean)
stats.register("Std", np.std)
stats.register("Min", np.min)
stats.register("Max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, 0.95, 0.2, 200, stats, halloffame=hof, verbose=1)

# Evaluate
y_train_pred, y_test_pred = make_predictions(str(hof[0]))

test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
acc_list.append(test_acc)
f1_list.append(test_f1)

print("Accuracy (test): {:.3f}".format(test_acc))
print("F1 score (test): {:.3f}".format(test_f1))

results = {"acc_list": acc_list,
           "f1_list": f1_list
           }

# Save results
output_path = "/workspace/results/ppsr_runs/{}_{}_run_{}.pkl".format(method_name, dataset_name, run_no)
with open(output_path, "wb") as f:
    pickle.dump(results, f)

print("Finished!")

