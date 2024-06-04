#!/bin/bash

local_dir="C:/Data/Projects/FedProM/Experiments/VFLBench/paper_experiments"  # e.g. "path_to_paper_experiments"
image_name="secretflow"

# Test method
method="p3ls"

# Define script path
script_path="/workspace/src/$method"

# List of dataset names
dataset_names=(
    "hysys_i"
    "hysys_ii"
    "sfs"
    "smp"
)

# Create the Docker container
container_name=$method

# Check if the container exists
if docker ps -a --format '{{.Names}}' | grep -q "^$container_name$"; then
  echo "Container '$container_name' already exists. Stopping and removing it..."
  # Stop the container
  docker stop "$container_name" >/dev/null
  # Remove the container
  docker rm "$container_name" >/dev/null
  echo "Container '$container_name' stopped and removed."
fi

docker run -dt --name $container_name -e PYTHONPATH="/workspace/src/p3ls/p3ls:$PYTHONPATH" -v "$local_dir:/workspace" $image_name

# Loop through each script
for dataset_name in "${dataset_names[@]}"; do
    echo "Test $dataset_name ..."
    docker exec -dt $container_name python "${script_path}/run_${method}_${dataset_name}.py"
done

## Clean up
#docker stop $container_name
#docker rm $container_name

echo "Press any key to exit..."
read -n 1 -s -r -p ""
echo "Exiting..."