# Import libraries
import pickle
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from p3ls.DataHolder import DataHolder
from p3ls.P3LS import P3LS
from p3ls.utils import generate_keys, generate_orthogonal_matrix

# Set random seeds
RANDOM_SEED = 1309
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load data
dataset_name = "hysys_i"
input_path = "/workspace/federated_data/{}.pkl".format(dataset_name)
with open(input_path, "rb") as f:
    fed_data = pickle.load(f)

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
Y_2_train = y_train
Y_2_test = y_test

# Preprocess data

# Define targets
n_targets = 1
Y_train = Y_2_train.reshape(-1, 1)
Y_test = Y_2_test.reshape(-1, 1)

# Data holder 1
x_1_scaler = StandardScaler()
X_1_train_scaled = x_1_scaler.fit_transform(X_1_train)
X_1_test_scaled = x_1_scaler.transform(X_1_test)

# Data holder 2
x_2_scaler = StandardScaler()
X_2_train_scaled = x_2_scaler.fit_transform(X_2_train)
X_2_test_scaled = x_2_scaler.transform(X_2_test)

y_scaler = StandardScaler()
Y_train_scaled = y_scaler.fit_transform(Y_train)

# Perform P3LS

# Create data holders

# Extract properties
m_train = X_1_train_scaled.shape[0]  # No. training samples
m_test = X_1_test_scaled.shape[0]  # No. validation samples
n_1 = X_1_train_scaled.shape[1]  # No. features of X1
n_2 = X_2_train_scaled.shape[1]  # No. features of X2
n_x = n_1 + n_2  # Total no. features
l = n_targets  # Total no. target variables

# Mask data
_, H_list = generate_keys(m=m_train, ns=[n_1, n_2], cache_dir="/workspace/src/p3ls/orthogonal_matrices", block_size=None)
H_1 = H_list[0]
H_2 = H_list[1]
G = generate_orthogonal_matrix(m=l)
A = generate_orthogonal_matrix(m=m_train)
N = np.random.rand(l, l)

# Assign data to data holders
data_holder_1 = DataHolder(name="passive_party_1", X=X_1_train_scaled, Y=None, A=A, H=H_1, G=None)
data_holder_2 = DataHolder(name="active_party", X=X_2_train_scaled, Y=Y_train_scaled, A=A, H=H_2, G=G)

# The CSP performs secure aggregation to get the masked data
X_train_encrypted = data_holder_1.X_encrypted + data_holder_2.X_encrypted
Y_train_encrypted = data_holder_2.Y_encrypted

# Run P3LS
n_runs = 20
rmse_list = []
r2_list = []

for i in range(n_runs):
    print("--- Run {}/{} ---".format(i + 1, n_runs))

    # The CSP trains the model
    model = P3LS(n_comp=10)
    model.fit(X_train_encrypted, Y_train_encrypted)

    # Recover results
    # Data holder 2 sends G_T_masked to the CSP: to replace G by a common key
    G_T_masked = G.T @ N
    model.set_g_t_masked(G_T_masked)

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

    # Data holders encrypt their predictions using a common random matrix/number
    C_train = np.random.rand(m_train, m_train)
    C_test = np.random.rand(m_test, m_test)
    C_train_inverse = np.linalg.pinv(C_train)
    C_test_inverse = np.linalg.pinv(C_test)

    Y_train_pred_scaled_1_enc = C_train @ Y_train_pred_scaled_1
    Y_test_pred_scaled_1_enc = C_test @ Y_test_pred_scaled_1
    Y_train_pred_scaled_2_enc = C_train @ Y_train_pred_scaled_2
    Y_test_pred_scaled_2_enc = C_test @ Y_test_pred_scaled_2

    # # The label holders requests the sum of local predictions
    Y_train_scaled_pred = C_train_inverse @ (
            Y_train_pred_scaled_1_enc + Y_train_pred_scaled_2_enc)
    Y_test_scaled_pred = C_test_inverse @ (
            Y_test_pred_scaled_1_enc + Y_test_pred_scaled_2_enc)

    # Convert predictions to original scale
    Y_train_pred = y_scaler.inverse_transform(Y_train_scaled_pred)
    Y_test_pred = y_scaler.inverse_transform(Y_test_scaled_pred)

    test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    test_r2 = r2_score(Y_test, Y_test_pred)
    rmse_list.append(test_rmse)
    r2_list.append(test_r2)

    print("R2 (test): {:.3f}".format(test_r2))
    print("RMSE (test): {:.3f}".format(test_rmse))

results = {"rmse_list": rmse_list,
           "r2_list": r2_list,
           }

# Save results
output_path = "/workspace/results/{}_{}.pkl".format(method_name, dataset_name)
with open(output_path, "wb") as f:
    pickle.dump(results, f)

print("Finished!")
