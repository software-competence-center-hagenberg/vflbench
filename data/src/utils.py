# Import libraries
import pandas as pd
import pickle
import tempfile
import os
import requests
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


def read_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            strings = content.split()
            return strings
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []


def vertical_split(x, num_parties):
    num_features = x.shape[1]
    xs = []
    for i in range(num_parties):
        if i == num_parties - 1:
            x_train_party_i = x[:, i * num_features // num_parties:]
        else:
            x_train_party_i = x[:, i * num_features // num_parties: (i + 1) * num_features // num_parties]
        xs.append(x_train_party_i)

    return xs


def load_data_cross_validation(x, y, num_parties=1, n_fold=5, split_indices=None, stratify=None, shuffle=True,
                               random_state=None):
    print("{} fold splitting".format(n_fold))
    results = []
    if n_fold > 1:
        if stratify is None:
            k_fold = KFold(n_splits=n_fold, shuffle=shuffle, random_state=random_state)
        else:
            k_fold = StratifiedKFold(n_splits=n_fold, shuffle=shuffle, random_state=random_state)
        for i, (train_idx, test_idx) in enumerate(k_fold.split(x, y)):
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_test = x[test_idx]
            y_test = y[test_idx]

            # Split data into parties
            if split_indices:
                xs_train = [x_train[:, split_indices[i]] for i in range(num_parties)]
                xs_test = [x_test[:, split_indices[i]] for i in range(num_parties)]
            else:
                xs_train = vertical_split(x_train, num_parties)
                xs_test = vertical_split(x_test, num_parties)

            results.append([xs_train, y_train, xs_test, y_test])
            print("Fold {} finished".format(i))
    else:  # fold = 1
        if split_indices:
            xs = [x[:, split_indices[i]] for i in range(num_parties)]
        else:
            xs = vertical_split(x, num_parties)
        results.append([xs, y])

    return results


def vertical_train_test_split(x, y, test_size, num_parties=1, split_indices=None, random_state=None, stratify=None,
                              shuffle=True):
    results = []

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=stratify, shuffle=shuffle,
                                                        random_state=random_state)

    # Split data into parties
    if split_indices:
        xs_train = [x_train[:, split_indices[i]] for i in range(num_parties)]
        xs_test = [x_test[:, split_indices[i]] for i in range(num_parties)]
    else:
        xs_train = vertical_split(x_train, num_parties)
        xs_test = vertical_split(x_test, num_parties)

    results.append([xs_train, y_train, xs_test, y_test])

    return results


def find_first_exceeding_index(array, threshold):
    indices = np.argwhere(array > threshold)
    if len(indices) > 0:
        return indices[0][0]  # Return the index of the first occurrence
    else:
        return None  # Return None if no value exceeds the threshold


def optimize_pca_n_comp(X, threshold):
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Maximum possible number of components
    max_n_comps = min(np.shape(X_scaled))

    model = PCA(n_components=max_n_comps)
    model.fit(X_scaled)

    cusum_explained_variance = np.cumsum(model.explained_variance_ratio_)

    best_n_comp = find_first_exceeding_index(cusum_explained_variance, threshold) + 1

    return best_n_comp


def download_zip(url, dest_folder):
    local_filename = url.split('/')[-1]
    zip_path = os.path.join(dest_folder, local_filename)

    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return zip_path


def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def delete_file(file_path):
    os.remove(file_path)


def download(url, dest_folder, extract_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    # Download the zip file
    zip_path = download_zip(url, dest_folder)

    # Unzip the file
    unzip_file(zip_path, extract_folder)

    # Delete the zip file
    delete_file(zip_path)

    print(f"Downloaded, extracted, and deleted the zip file: {zip_path}")
