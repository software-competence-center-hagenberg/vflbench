# Import libraries
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from utils import vertical_train_test_split, optimize_pca_n_comp
from utils import download

# Set random seed
RANDOM_SEED = 1309
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Download the dataset at: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
# Comment the following lines if the dataset has been downloaded and stored in /data/original_dataset/

# url = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
# dest_folder = "../original_data/"  # Destination folder for the zip file
# extract_folder = "../original_data/hydraulic_system"  # Folder to extract contents to
# download(url, dest_folder, extract_folder)

# Load data
stage_1_file_names = ["PS4", "PS5", "PS6", "TS3", "TS4", "FS2", "CE", "CP"]
stage_2_file_names = ["EPS1", "FS1", "PS1", "PS2", "PS3", "TS1", "TS2", "VS1", "SE"]

stage_1_data = [np.genfromtxt("../original_data/hydraulic_system/{}.txt".format(name)) for name in stage_1_file_names]
stage_2_data = [np.genfromtxt("../original_data/hydraulic_system/{}.txt".format(name)) for name in stage_2_file_names]

targets = np.genfromtxt("../original_data/hydraulic_system/profile.txt")

# Assign data to different data holders
X_1 = np.concatenate(stage_1_data, axis=1)
X_2 = np.concatenate(stage_2_data, axis=1)

# Concatenate data
X = np.concatenate([X_1, X_2], axis=1)
y = targets[:, [4]]

# Assign features to different parties
split_indices = [list(range(X_1.shape[1])), list(range(X_1.shape[1], X.shape[1]))]

# Perform train-test split
num_parties = 2
cross_valid_data = vertical_train_test_split(x=X, y=y, test_size=0.2, num_parties=num_parties,
                                             split_indices=split_indices, random_state=RANDOM_SEED)

# Apply PCA
Xs_train, y_train, Xs_test, y_test = cross_valid_data[0]

# Find the optimal no. PCs
threshold = 0.975
n_comp_1 = optimize_pca_n_comp(Xs_train[0], threshold)
n_comp_2 = optimize_pca_n_comp(Xs_train[1], threshold)
print("No. comp 1: ", n_comp_1)
print("No. comp 2: ", n_comp_2)

pipeline_1 = Pipeline([("prior_scaling", StandardScaler()),
                       ("pca", PCA(n_components=n_comp_1))])

pipeline_2 = Pipeline([("prior_scaling", StandardScaler()),
                       ("pca", PCA(n_components=n_comp_2))])

X_1_train_transformed = pipeline_1.fit_transform(Xs_train[0])
X_1_test_transformed = pipeline_1.transform(Xs_test[0])
X_2_train_transformed = pipeline_2.fit_transform(Xs_train[1])
X_2_test_transformed = pipeline_2.transform(Xs_test[1])

Xs_train_preprocessed = [X_1_train_transformed, X_2_train_transformed]
Xs_test_preprocessed = [X_1_test_transformed, X_2_test_transformed]

cross_valid_data_preprocessed = [Xs_train_preprocessed, y_train, Xs_test_preprocessed, y_test]

# Save data
dataset_name = "hysys_ii"
fed_data = {
    "num_parties": num_parties,
    "task": "regression",
    "cross_valid_data": [cross_valid_data_preprocessed]
}

output_path = "../federated_data/{}.pkl".format(dataset_name)
with open(output_path, "wb") as f:
    pickle.dump(fed_data, f)

