# Import libraries
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import secretflow as sf
from tensorflow import keras
from tensorflow.keras import layers
from secretflow.ml.nn import SLModel
from secretflow.security.privacy import DPStrategy, LabelDP
from secretflow.security.privacy.mechanism.tensorflow import GaussianEmbeddingDP
from secretflow.data import FedNdarray, PartitionWay
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set random seed
RANDOM_SEED = 1309
tf.keras.utils.set_random_seed(RANDOM_SEED)
tf.config.experimental.enable_op_determinism()

# Initialize SF
sf.init(["c_1", "c_2"], address="local")
c_1, c_2 = sf.PYU("c_1"), sf.PYU("c_2")
spu = sf.SPU(sf.utils.testing.cluster_def(["c_1", "c_2"]))

# Load data
dataset_name = "sfs"
input_path = "/workspace/federated_data/{}.pkl".format(dataset_name)
with open(input_path, "rb") as f:
    fed_data = pickle.load(f)

cross_valid_data = fed_data["cross_valid_data"]
n_clients = fed_data["num_parties"]
task = fed_data["task"]
method_name = "splitnn"

Xs_train, y_train_pt, Xs_test, y_test_pt = cross_valid_data[0]

# Prepare data
m = Xs_train[0].shape[0]
n_c_1 = Xs_train[0].shape[1]
n_c_2 = Xs_train[1].shape[1]


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

# Encrypt data
X_train = FedNdarray({globals()["c_{}".format(i)]: (globals()["c_{}".format(i)](lambda: Xs_train[i - 1])()) for i in
                      range(1, n_clients + 1)},
                     partition_way=PartitionWay.VERTICAL,
                     )

y_train = FedNdarray(
    {globals()["c_{}".format(n_clients)]: (globals()["c_{}".format(n_clients)](lambda: y_train_pt)())},
    partition_way=PartitionWay.VERTICAL,
)

X_test = FedNdarray({globals()["c_{}".format(i)]: (globals()["c_{}".format(i)](lambda: Xs_test[i - 1])()) for i in
                     range(1, n_clients + 1)},
                    partition_way=PartitionWay.VERTICAL,
                    )

y_test = FedNdarray(
    {globals()["c_{}".format(n_clients)]: (globals()["c_{}".format(n_clients)](lambda: y_test_pt)())},
    partition_way=PartitionWay.VERTICAL,
)


# Prepare model
def create_base_model(input_dim, output_dim, name="base_model"):
    # Create model
    def create_model():
        model = keras.Sequential(
            [
                keras.Input(shape=input_dim),
                layers.Dense(32, activation="relu"),
                layers.Dense(output_dim, activation="relu"),
            ]
        )

        # Compile model
        model.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=["mse"],
        )

        return model

    return create_model


def create_fuse_model(input_dim, output_dim, party_nums, name="fuse_model"):
    def create_model():
        input_layers = []
        for i in range(party_nums):
            input_layers.append(
                keras.Input(
                    input_dim,
                )
            )

        merged_layer = layers.concatenate(input_layers)
        fuse_layer_1 = layers.Dense(32, activation="relu")(merged_layer)
        fuse_layer_2 = layers.Dense(16, activation="relu")(fuse_layer_1)
        fuse_layer_3 = layers.Dense(8, activation="relu")(fuse_layer_2)
        output = layers.Dense(output_dim)(fuse_layer_3)

        model = keras.Model(inputs=input_layers, outputs=output)

        # Compile model
        model.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=["mse"],
        )
        return model

    return create_model


hidden_size = 32
model_base_c_1 = create_base_model(n_c_1, hidden_size)
model_base_c_2 = create_base_model(n_c_2, hidden_size)
base_model_dict = {c_1: model_base_c_1, c_2: model_base_c_2}
model_fuse = create_fuse_model(input_dim=hidden_size, party_nums=n_clients, output_dim=1)

# Define DP operations
batch_size = 10

gaussian_embedding_dp = GaussianEmbeddingDP(
    noise_multiplier=0.005,
    l2_norm_clip=1.0,
    batch_size=batch_size,
    num_samples=m,
    is_secure_generator=False,
)
# label_dp = LabelDP(eps=64.0)  # The current version of Secretflow doesn't support continuous targets

dp_strategy_c_1 = DPStrategy(embedding_dp=gaussian_embedding_dp)
dp_strategy_c_2 = DPStrategy(embedding_dp=gaussian_embedding_dp)
# dp_strategy_c_2 = DPStrategy(label_dp=label_dp)
dp_strategy_dict = {c_1: dp_strategy_c_1, c_2: dp_strategy_c_2}  # {c_1: dp_strategy_c_1}
dp_spent_step_freq = 10

# Define model
sl_model = SLModel(
    base_model_dict=base_model_dict,
    device_y=c_2,
    model_fuse=model_fuse,
    dp_strategy_dict=dp_strategy_dict,
)

# Run SplitNN
n_runs = 20
rmse_list = []
r2_list = []
train_time_list = []
test_time_list = []

for i in range(n_runs):
    print("--- Run {}/{} ---".format(i + 1, n_runs))

    # Train model
    history = sl_model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_freq=1,
        dp_spent_step_freq=dp_spent_step_freq
    )

    # Evaluate model
    y_test_pred_enc = sl_model.predict(X_test)
    y_test_pred = sf.reveal(y_test_pred_enc)
    y_test_pred = tf.concat(y_test_pred, axis=0).numpy()

    test_rmse = np.sqrt(mean_squared_error(y_test_pt, y_test_pred))
    test_r2 = r2_score(y_test_pt, y_test_pred)
    rmse_list.append(test_rmse)
    r2_list.append(test_r2)

    print("R2 (test): {:.3f}".format(test_r2))
    print("RMSE (test): {:.3f}".format(test_rmse))

# Save results
results = {"rmse_list": rmse_list,
           "r2_list": r2_list
           }

# Save results
output_path = "/workspace/results/{}_{}.pkl".format(method_name, dataset_name)
with open(output_path, "wb") as f:
    pickle.dump(results, f)

print("Finished!")
