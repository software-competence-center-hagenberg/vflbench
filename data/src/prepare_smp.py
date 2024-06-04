# Import libraries
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import vertical_train_test_split

# Set random seed
RANDOM_SEED = 1309
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Load data
data_folder = "../original_data/simulated_multistage_process/"
data = pickle.load(open(data_folder + "fed_sim_exp_data.pkl", "rb"))

n_datasets = data["n_datasets"]
exp_data = data["exp_1"]
dataset_no = 1
selected_target_idx = 0

# Extract data
fed_data = exp_data["dataset_{}".format(dataset_no)]
n_parties = fed_data["n_stages"]

# Data holder 1
X_1_train = fed_data["stage_1"]["X_train"]
X_1_val = fed_data["stage_1"]["X_val"]
X_1_test = fed_data["stage_1"]["X_test"]

# Data holder 2
X_2_train = fed_data["stage_2"]["X_train"]
X_2_val = fed_data["stage_2"]["X_val"]
X_2_test = fed_data["stage_2"]["X_test"]

# Data holder 3
X_3_train = fed_data["stage_3"]["X_train"]
X_3_val = fed_data["stage_3"]["X_val"]
X_3_test = fed_data["stage_3"]["X_test"]

y_train = fed_data["stage_3"]["Y_train"][:, selected_target_idx]
y_val = fed_data["stage_3"]["Y_val"][:, selected_target_idx]
y_test = fed_data["stage_3"]["Y_test"][:, selected_target_idx]

# Merge X
X_train = np.concatenate([X_1_train, X_2_train, X_3_train], axis=1)
X_val = np.concatenate([X_1_val, X_2_val, X_3_val], axis=1)
X_test = np.concatenate([X_1_test, X_2_test, X_3_test], axis=1)

X = np.concatenate([X_train, X_val, X_test], axis=0)
y = np.concatenate([y_train, y_val, y_test], axis=0)

feature_indices = list(range(X.shape[1]))
split_indices = [feature_indices[:X_1_train.shape[1]],
                 feature_indices[X_1_train.shape[1]:(X_1_train.shape[1] + X_2_train.shape[1])],
                 feature_indices[(X_1_train.shape[1] + X_2_train.shape[1]):]
                 ]

# Perform train-test split
num_parties = 3
cross_valid_data = vertical_train_test_split(x=X, y=y, test_size=0.2, num_parties=num_parties,
                                             split_indices=split_indices, random_state=RANDOM_SEED)

# Save data
dataset_name = "smp"
fed_data = {
    "num_parties": num_parties,
    "task": "regression",
    "cross_valid_data": cross_valid_data
}

output_path = "../federated_data/{}.pkl".format(dataset_name)
with open(output_path, "wb") as f:
    pickle.dump(fed_data, f)
