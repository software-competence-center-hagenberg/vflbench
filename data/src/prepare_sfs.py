# Import libraries
import pandas as pd
import numpy as np
import random
import pickle
from utils import vertical_train_test_split

# Set random seed
RANDOM_SEED = 1309
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Load data
# Download the original dataset beforehand
df = pd.read_csv("../original_data/steel_fatigue_strength/data.csv")

# Split data
X = df.drop(columns=["Sl. No.", "Fatigue"])
y = df["Fatigue"].values.reshape(-1, 1)

chem_comp = ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo"]
upstream = ["RedRatio", "dA", "dB", "dC"]
heat_treatment = ["NT", "THT", "THt", "THQCr", "CT", "Ct", "DT", "Dt", "QmT", "TT", "Tt", "TCr"]

manu_cols = upstream + heat_treatment

X_1 = X.drop(columns=manu_cols).values
X_2 = X[manu_cols].values

# Concatenate data
X = np.concatenate([X_1, X_2], axis=1)

# Assign features to different parties
split_indices = [list(range(X_1.shape[1])), list(range(X_1.shape[1], X.shape[1]))]

# Perform train-test split
num_parties = 2
cross_valid_data = vertical_train_test_split(x=X, y=y, test_size=0.2, num_parties=num_parties,
                                             split_indices=split_indices, random_state=RANDOM_SEED)

# Save data
dataset_name = "sfs"
fed_data = {
    "num_parties": num_parties,
    "task": "regression",
    "cross_valid_data": cross_valid_data
}

output_path = "../federated_data/{}.pkl".format(dataset_name)
with open(output_path, "wb") as f:
    pickle.dump(fed_data, f)

