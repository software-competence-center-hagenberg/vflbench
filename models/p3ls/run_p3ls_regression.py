# Import libraries
import pickle
import random
import time
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from fedpls.DataHolder import DataHolder
from fedpls.FedPLS import FedPLS
from fedpls.utils import generate_keys, generate_orthogonal_matrix
from utils import read_pickle, save_pickle, load_data

# Set random seeds
RANDOM_SEED = 1309
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load data
data_version = "v4"
data_source = "natural_split_datasets"
dataset_name = "simulated_multistage_process"
fed_data = load_data(data_version, data_source, dataset_name)
cross_valid_data = fed_data["cross_valid_data"]
n_clients = fed_data["num_parties"]
task = fed_data["task"]
method_name = "p3ls"

Xs_train, y_train, Xs_test, y_test = cross_valid_data[0]

# Data holder 1
X_1_train = Xs_train[0]
X_1_test = Xs_test[0]

# Data holder 2
X_2_train = Xs_train[1]
X_2_test = Xs_test[1]

# Data holder 3
X_3_train = Xs_train[2]
X_3_test = Xs_test[2]
Y_3_train = y_train
Y_3_test = y_test

# Preprocess data

# Define targets
n_targets = 1
Y_train = Y_3_train.reshape(-1, 1)
Y_test = Y_3_test.reshape(-1, 1)


# Data holder 1
x_1_scaler = StandardScaler()
X_1_train_scaled = x_1_scaler.fit_transform(X_1_train)
X_1_test_scaled = x_1_scaler.transform(X_1_test)

# Data holder 2
x_2_scaler = StandardScaler()
X_2_train_scaled = x_2_scaler.fit_transform(X_2_train)
X_2_test_scaled = x_2_scaler.transform(X_2_test)

# Data holder 3
x_3_scaler = StandardScaler()
X_3_train_scaled = x_3_scaler.fit_transform(X_3_train)
X_3_test_scaled = x_3_scaler.transform(X_3_test)

y_scaler = StandardScaler()
Y_train_scaled = y_scaler.fit_transform(Y_train)

# Perform P3LS

# Create data holders

# Extract properties
m_train = X_1_train_scaled.shape[0]  # No. training samples
m_test = X_1_test_scaled.shape[0]  # No. validation samples
n_1 = X_1_train_scaled.shape[1]  # No. features of X1
n_2 = X_2_train_scaled.shape[1]  # No. features of X2
n_3 = X_3_train_scaled.shape[1]  # No. features of X3
n_x = n_1 + n_2 + n_3  # Total no. features
l = n_targets  # Total no. target variables

# Mask data
_, H_list = generate_keys(m=m_train, ns=[n_1, n_2, n_3], cache_dir="./orthogonal_matrices", block_size=None)
H_1 = H_list[0]
H_2 = H_list[1]
H_3 = H_list[2]
G = generate_orthogonal_matrix(m=l)
A = generate_orthogonal_matrix(m=m_train)
N = np.random.rand(l, l)

# Assign data to data holders
data_holder_1 = DataHolder(name="passive_party_1", X=X_1_train_scaled, Y=None, A=A, H=H_1, G=None)
data_holder_2 = DataHolder(name="passive_party_2", X=X_2_train_scaled, Y=None, A=A, H=H_2, G=None)
data_holder_3 = DataHolder(name="active_party", X=X_3_train_scaled, Y=Y_train_scaled, A=A, H=H_3, G=G)

# The CSP performs secure aggregation to get the masked data
X_train_encrypted = data_holder_1.X_encrypted + data_holder_2.X_encrypted + data_holder_3.X_encrypted
Y_train_encrypted = data_holder_3.Y_encrypted

# Run P3LS
n_runs = 20
rmse_list = []
r2_list = []
train_time_list = []
test_time_list = []

for i in range(n_runs):
    print("--- Run {}/{} ---".format(i + 1, n_runs))

    # The CSP trains the model
    model = FedPLS(n_comp=10)
    model.fit(X_train_encrypted, Y_train_encrypted)

    # Recover results
    # Data holder 3 sends G_T_masked to the CSP: to replace G by a common key
    G_T_masked = G.T @ N
    model.set_g_t_masked(G_T_masked)

    # Data holder 3
    coef_3_masked = model.get_coef_masked(data_holder_3.H_encrypted)
    coef_3 = data_holder_3.recover_coef(coef_3_masked, N)

    # Data holder 1
    coef_1_masked = model.get_coef_masked(data_holder_1.H_encrypted)
    coef_1 = data_holder_1.recover_coef(coef_1_masked, N)

    # Data holder 2
    coef_2_masked = model.get_coef_masked(data_holder_2.H_encrypted)
    coef_2 = data_holder_2.recover_coef(coef_2_masked, N)

    # Make predictions on training and validation set

    # Data holders make local predictions
    Y_train_pred_scaled_1 = data_holder_1.predict(X_1_train_scaled)
    Y_test_pred_scaled_1 = data_holder_1.predict(X_1_test_scaled)

    Y_train_pred_scaled_2 = data_holder_2.predict(X_2_train_scaled)
    Y_test_pred_scaled_2 = data_holder_2.predict(X_2_test_scaled)

    Y_train_pred_scaled_3 = data_holder_3.predict(X_3_train_scaled)
    Y_test_pred_scaled_3 = data_holder_3.predict(X_3_test_scaled)

    # Data holders encrypt their predictions using a common random matrix/number
    C_train = np.random.rand(m_train, m_train)
    C_test = np.random.rand(m_test, m_test)
    C_train_inverse = np.linalg.pinv(C_train)
    C_test_inverse = np.linalg.pinv(C_test)

    Y_train_pred_scaled_1_enc = C_train @ Y_train_pred_scaled_1
    Y_test_pred_scaled_1_enc = C_test @ Y_test_pred_scaled_1
    Y_train_pred_scaled_2_enc = C_train @ Y_train_pred_scaled_2
    Y_test_pred_scaled_2_enc = C_test @ Y_test_pred_scaled_2
    Y_train_pred_scaled_3_enc = C_train @ Y_train_pred_scaled_3
    Y_test_pred_scaled_3_enc = C_test @ Y_test_pred_scaled_3

    # # The label holders requests the sum of local predictions
    Y_train_scaled_pred = C_train_inverse @ (
            Y_train_pred_scaled_1_enc + Y_train_pred_scaled_2_enc + Y_train_pred_scaled_3_enc)
    Y_test_scaled_pred = C_test_inverse @ (
            Y_test_pred_scaled_1_enc + Y_test_pred_scaled_2_enc + Y_test_pred_scaled_3_enc)

    # Convert predictions to original scale
    Y_train_pred = y_scaler.inverse_transform(Y_train_scaled_pred)
    Y_test_pred = y_scaler.inverse_transform(Y_test_scaled_pred)

    train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
    train_r2 = r2_score(Y_train, Y_train_pred)

    test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    test_r2 = r2_score(Y_test, Y_test_pred)
    rmse_list.append(test_rmse)
    r2_list.append(test_r2)

    print("R2 (test): {:.3f}".format(test_r2))
    print("RMSE (test): {:.3f}".format(test_rmse))

results = {"rmse_list": rmse_list,
           "r2_list": r2_list,
           "train_time_list": train_time_list,
           "test_time_list": test_time_list
           }

save_pickle(results,
            "vflbench/results_{}/{}/fed_models/{}_{}.pkl".format(data_version, data_source, method_name, dataset_name))

print("Finished!")


