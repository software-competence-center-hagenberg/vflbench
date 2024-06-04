#!/bin/bash

local_dir="path_to_paper_experiments"  # e.g. C:/Data/Projects/FedProM/Experiments/VFLBench/paper_experiments
image_name="secretflow"

# Test method
method="ppsr"

# Define script path
script_path="/workspace/src/$method"

# Define result directories
ind_result_dir="/workspace/results/ppsr_runs"
agg_result_dir="/workspace/results"

# List of dataset names
dataset_names=(
    "hysys_i"
    "hysys_ii"
    "sfs"
    "smp"
)

n_runs=20

# Loop through each script
for dataset_name in "${dataset_names[@]}"; do
  echo "Test $dataset_name ..."

  # Create the Docker container
  container_name="${method}_$dataset_name"

  # Check if the container exists
  if docker ps -a --format '{{.Names}}' | grep -q "^$container_name$"; then
    echo "Container '$container_name' already exists. Stopping and removing it..."
    # Stop the container
    docker stop "$container_name" >/dev/null
    # Remove the container
    docker rm "$container_name" >/dev/null
    echo "Container '$container_name' stopped and removed."
  fi

  docker run --shm-size=32gb -dt --name "$container_name" -e PYTHONPATH="/workspace/src/ppsr/ppsr:$PYTHONPATH" -v "$local_dir:/workspace" $image_name

  # Run the algorithm multiple times
  for i in $(seq 1 $n_runs)
  do
    echo "Run $i / $n_runs"
    docker exec "$container_name" python "${script_path}/run_${method}_${dataset_name}.py" -r "$i"
  done

  # Merge results of multiple runs
  docker exec "$container_name" python "${script_path}/aggregate_results.py" -ird $ind_result_dir -mrd $agg_result_dir -m $method -d "$dataset_name" -nr $n_runs

#  # Clean up
#  docker stop "$container_name"
#  docker rm "$container_name"
done

echo "Press any key to exit..."
read -n 1 -s -r -p ""
echo "Exiting..."