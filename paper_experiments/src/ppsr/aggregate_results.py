# Import libraries
import pickle
import argparse

# Extract arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ind_result_dir", "-ird", help="Individual result directory")
parser.add_argument("--merged_result_dir", "-mrd", help="Merged result directory")
parser.add_argument("--method", "-m", help="Method name")
parser.add_argument("--dataset", "-d", help="Dataset name")
parser.add_argument("--n_runs", "-nr", help="Run no.")
args = parser.parse_args()

# ind_result_dir = args.ind_result_dir
# merged_result_dir = args.merged_result_dir
# method = args.method
# dataset = args.dataset
# n_runs = int(args.n_runs)

# Test
ind_result_dir = "C:/Data/Projects/FedProM/Experiments/VFLBench/paper_experiments/results/ppsr_runs"
merged_result_dir = "C:/Data/Projects/FedProM/Experiments/VFLBench/paper_experiments/results"
method = "ppsr"
dataset = "smp"
n_runs = 20


# Aggregate results from individual runs into a single file
rmse_list = []
r2_list = []
acc_list = []
f1_list = []

task = "regression"
for i in range(1, n_runs + 1):

    file_path = "{}/{}_{}_run_{}.pkl".format(ind_result_dir, method, dataset, i)
    try:

        with open(file_path, "rb") as f:
            result = pickle.load(f)

        if "rmse_list" in result:
            rmse_list.append(result["rmse_list"][0])
            r2_list.append(result["r2_list"][0])
        else:
            task = "classification"
            acc_list.append(result["acc_list"][0])
            f1_list.append(result["f1_list"][0])
    except:
        print("{} doesn't exit".format(file_path))

if task == "regression":
    merged_results = {"rmse_list": rmse_list, "r2_list": r2_list}
else:
    merged_results = {"acc_list": acc_list, "f1_list": f1_list}

# Save results
output_path = "{}/{}_{}.pkl".format(merged_result_dir, method, dataset)

with open(output_path, "wb") as f:
    pickle.dump(merged_results, f)
