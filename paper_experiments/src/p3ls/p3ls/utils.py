import numpy as np
import os
import pickle
import scipy.stats


def generate_orthogonal_matrix(m):
    """
    Generate a random orthogonal matrix

    Parameters
    ----------
    m: integer
        No. rows/columns

    Returns
    -------
    q: numpy array of shape (m, m)
        A random orthogonal matrix
    """
    q, _ = np.linalg.qr(np.random.randn(m, m), mode="full")

    return q


def generate_orthogonal_matrix_efficient(n, orthogonal_matrix_cache_dir, reuse=False, block_size=None):
    """
    Efficient method for generating orthogonal matrices

    Parameters
    ----------
    n: int
        No. columns/rows

    orthogonal_matrix_cache_dir: string
        Path to the cache directory

    reuse: boolean
        Whether to reuse the building blocks

    block_size: int
        Block size

    Returns
    -------
    q: numpy array of shape (n, n)
        A random orthogonal matrix

    """
    if os.path.isdir(orthogonal_matrix_cache_dir) is False:
        os.makedirs(orthogonal_matrix_cache_dir, exist_ok=True)
    file_list = os.listdir(orthogonal_matrix_cache_dir)
    existing = [e.split('.')[0] for e in file_list]

    file_name = str(n)
    if block_size is not None:
        file_name += '_blc%s' % block_size

    if reuse and file_name in existing:
        with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'rb') as file:
            return pickle.load(file)
    else:
        if block_size is not None:
            qs = [block_size] * int(n / block_size)
            if n % block_size != 0:
                qs[-1] += (n - np.sum(qs))
            q = np.zeros([n, n])
            for i in range(len(qs)):
                sub_n = qs[i]
                tmp = generate_orthogonal_matrix(sub_n)
                index = int(np.sum(qs[:i]))
                q[index:index + sub_n, index:index + sub_n] += tmp
        else:
            q = generate_orthogonal_matrix(n)
        if reuse:
            with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'wb') as file:
                pickle.dump(q, file, protocol=4)
        return q


def generate_keys(m, ns, cache_dir, block_size=None, reuse=False):
    """
    Generate keys for encrypting private data and recovering the results

    Parameters
    ----------

    m: integer:
        No. rows of the dataset

    ns: list
        A list of no. columns of each client's data

    cache_dir: string
        Catch directory

    block_size: int/None
        Number of building blocks used in the orthogonal matrix generator

    Returns
    -------
    P: numpy array of shape (m, m)
        A random orthogonal matrix

    Qi_list: list
        A list of numpy arrays of shape (ni, n), where ni is the no. variables of data holder ith 's dataset,
        and n is the total no. variables of all data holders

    """

    k = len(ns)  # No. data holders

    n = sum(ns)  # Total number of variables

    # Generate P
    P = generate_orthogonal_matrix_efficient(n=m, orthogonal_matrix_cache_dir=cache_dir, reuse=reuse,
                                             block_size=block_size)

    # Generate Q
    Q = generate_orthogonal_matrix_efficient(n=n, orthogonal_matrix_cache_dir=cache_dir, reuse=reuse,
                                             block_size=block_size)

    # Split Q horizontally into multiple blocks with each block corresponds to one data holder
    Qi_list = []

    prev_idx = 0
    for i in range(k):
        ni = ns[i]
        Qi = Q[prev_idx:(prev_idx + ni), :]
        Qi_list.append(Qi)

        prev_idx = prev_idx + ni

    return P, Qi_list


def generate_rand_square_mat(m):
    """
    Generate a random square matrix

    Parameters
    ----------
    m: integer
        No. rows of the square matrix

    Returns
    -------
    rand_square_mat: numpy array of shape (m, m)
        A randomly generated square matrix

    """

    rand_square_mat = np.random.rand(m, m)

    return rand_square_mat


def mask_data(X, P, Q):
    """
    Mask data using two random orthogonal matrices

    Parameters
    ----------
    X: numpy array of shape (m, n)
        Raw data to be masked

    P: numpy array of shape (m, m)
        The left random orthogonal matrix

    Q: numpy array of shape (n, n)
        The right random orthogonal matrix

    Returns
    -------
    X_masked: numpy array of shape (m, n)
        Encrypted data

    """

    X_masked = P @ X @ Q
    return X_masked


def mask_Qi_T(Qi_T, Ri):
    """
    Mask the matrix Qi using a random square matrix

    Parameters
    ----------
    Qi_T: numpy array of shape (n, ni)
        The matrix to be masked

    Ri: numpy array of shape (ni, ni)
        A random square matrix

    Returns
    -------
    Qi_T_masked: numpy array of shape (n, ni)
        The masked matrix
    """

    Qi_T_masked = Qi_T @ Ri
    return Qi_T_masked


def mask_scores(scores, P):
    scores_masked = P @ scores
    return scores_masked


def transform(scores_list):
    """
    Another way to calculate the scores

    This function can work even in the online mode

    Parameters
    ----------
    scores_list: list
        A list of masked scores

    Returns
    -------
    T_masked : numpy array of shape (m, n_comp)
        Aggregated scores
    """
    # Concatenate data
    T_masked = np.sum(scores_list, axis=0)

    return T_masked


def _svd_flip_1d(u, v):
    """Same as svd_flip but works on 1d arrays, and is inplace"""
    # svd_flip would force us to convert to 2d array and would also return 2d
    # arrays. We don't want that.
    biggest_abs_val_idx = np.argmax(np.abs(u))
    sign = np.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign


def encrypt_local_key(Mi, G, m, start, end):
    """
    Encrypt local keys (left key and right key)

    Parameters
    ----------
    Mi: numpy array
        The left key
    G: numpy array
        The key for masking M
    m: int
        The number of columns of the left matrix after being masked
    start: int
        The starting index of Mi
    end: int
        The ending index of Mi

    Returns
    -------
    M_masked: numpy array

    """

    # Create M_masked
    M = np.zeros((m, m))
    M[start:end, :] = Mi
    M_masked = G @ M
    M_T_masked = M.T @ np.linalg.inv(G)

    return M_masked, M_T_masked


