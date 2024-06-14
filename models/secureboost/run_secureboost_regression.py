# Import libraries
import timeit
import spu
import pickle
import argparse
import random
import numpy as np
import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal, wait
from secretflow.ml.boost.sgb_v import Sgb, get_classic_XGB_params
from secretflow.ml.boost.sgb_v.model import load_model
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed
RANDOM_SEED = 1309
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load data
# Read Xs_train, y_train, Xs_test, y_test

# Set up the devices
_system_config = {"lineage_pinning_enabled": False}
sf.shutdown()

# Init cluster
sf.init(
    ["c_{}".format(i) for i in range(1, n_clients + 1)],
    address="local",
    _system_config=_system_config,
    object_store_memory=5 * 1024 * 1024 * 1024,
)

# SPU settings
cluster_def = {
    "nodes": [{"party": "c_{}".format(i), "id": "local:{}".format(i - 1), "address": "127.0.0.1:{}".format(12945 + i)}
              for i in range(1, n_clients + 1)],
    "runtime_config": {
        # SEMI2K support 2/3 PC, ABY3 only support 3PC, CHEETAH only support 2PC.
        # pls pay attention to size of nodes above. nodes size need match to PC setting.
        "protocol": spu.spu_pb2.SEMI2K,
        "field": spu.spu_pb2.FM128,
    },
}

# HEU settings
heu_config = {
    "evaluators": [{"party": "c_{}".format(i)} for i in range(1, n_clients)],
    "sk_keeper": {"party": "c_{}".format(n_clients)},
    "mode": "PHEU",
    "he_parameters": {
        # ou is a fast encryption schema that is as secure as paillier.
        "schema": "ou",
        "key_pair": {
            "generate": {
                # bit size should be 2048 to provide sufficient security.
                "bit_size": 2048,
            },
        },
    },
    "encoding": {
        "cleartext_type": "DT_I32",
        "encoder": "IntegerEncoder",
        "encoder_args": {"scale": 1},
    },
}

for i in range(1, n_clients + 1):
    globals()["c_{}".format(i)] = sf.PYU("c_{}".format(i))

heu = sf.HEU(heu_config, cluster_def["runtime_config"]["field"])

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

# Prepare Params
params = get_classic_XGB_params()
params["num_boost_round"] = 200
params["learning_rate"] = 0.1
params["rowsample_by_tree"] = 0.8
params["colsample_bytree"] = 0.8
params["max_depth"] = 3
params["objective"] = "linear"
# pp.pprint(params)

# Run Sgb
n_runs = 20
rmse_list = []
r2_list = []

for i in range(n_runs):
    print("--- Run {}/{} ---".format(i + 1, n_runs))

    sgb = Sgb(heu)

    train_st = timeit.default_timer()
    model = sgb.train(params, X_train, y_train)
    train_et = timeit.default_timer()
    train_time_list.append(train_et - train_st)

    # Model Evaluation
    test_st = timeit.default_timer()
    y_test_pred = model.predict(X_test)
    y_test_pred_pt = reveal(y_test_pred)  # Decrypt predictions
    test_et = timeit.default_timer()
    test_time_list.append(test_et - test_st)

    test_rmse = np.sqrt(mean_squared_error(y_test_pt, y_test_pred_pt))
    test_r2 = r2_score(y_test_pt, y_test_pred_pt)
    rmse_list.append(test_rmse)
    r2_list.append(test_r2)

    print("R2 (test): {:.3f}".format(test_r2))
    print("RMSE (test): {:.3f}".format(test_rmse))

results = {"rmse_list": rmse_list,
           "r2_list": r2_list,
           "train_time_list": train_time_list,
           "test_time_list": test_time_list
           }

# Save results

print("Finished!")
