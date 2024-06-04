import numpy as np
import os
import pickle
import scipy.stats
import matplotlib.pyplot as plt


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


def apply_efficient_masking(X, P, block_size):
    pass


def apply_efficient_recovery(Q, R, block_size):
    pass


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


def calculate_T2(t_new, eigenvalues):
    """
    Calculate Hotelling's T2

    Parameters
    ----------
    eigenvalues: numpy array of shape (n_comp,)
        The eigenvalues associated with the retained principal components

    t_new: numpy array of shape
        Scores of the sample

    Returns
    -------
    T_new_2: numpy array of shape
        Hotelling's T2 statistic

    References
    ----------
    
    Statistical process monitoring with independent component analysis - Lee et al. - 2004
    

    """

    T_new_2 = t_new @ (np.diag(eigenvalues ** -1)) @ t_new.T

    return T_new_2


def calculate_T2_contributions(scores, eigenvalues, loadings):
    """
    Calculate T2 contributions

    Parameters
    ----------
    scores: numpy array of shape (1, n_comp)
        Scores of the sample after being projected on the model

    eigenvalues: numpy array of shape (n_comp, )
        Eigenvalues of the model

    loadings

    Returns
    -------
    T2_cont: numpy array of shape (1, ni)
        T2 contributions
        ni is the number of variables that the participant have


    References
    ----------

    https://wiki.eigenvector.com/index.php?title=T-Squared_Q_residuals_and_Contributions

    """
    T2_cont = scores @ np.diag(eigenvalues ** -1 / 2) @ loadings.T
    return T2_cont


def calculate_T2_time_interval_contributions(T2_cont, n_vars):
    """
    Calculate contributions of time intervals

    Parameters
    ----------
    T2_cont:
    n_vars:

    Returns
    -------

    """
    T2_time_interval_cont = np.sum(np.reshape(T2_cont, (n_vars, -1), order='F'), axis=0)

    return T2_time_interval_cont


def calculate_T2_limit(n_batches, n_comp, alpha):
    """
    Calculate confidence intervals on Hotelling's T2

    For more details, see:
    Statistical process monitoring with independent component analysis, Lee et al., 2004

    Parameters
    ----------
    n_batches: int
        Number of NOC batches

    n_comp: int
        Number of selected principal components

    alpha: float
        Significance level


    Returns
    -------
    T2_alpha: float
        The upper confidence limit for Hotelling's T2

    """
    T2_alpha = (n_comp * (n_batches - 1) / (n_batches - n_comp)) * scipy.stats.f.ppf(1 - alpha, n_comp,
                                                                                     n_batches - n_comp)
    return T2_alpha


def calculate_Q_limit(eig_residuals, alpha):
    """
    Q-Residuals Limits Calculated by the Jackson-Mudholkar Approximation

    Compute Residual Limits (Confidence Intervals) following the Jackson-Mudholkar approximation
    Jackson, J.E., "A User's Guide to Principal Components", John Wiley & Sons, New York, NY (1991)

    Parameters
    ----------

    eig_residuals: numpy array (n-n_comp, )
        Eigenvalues corresponding to unused principal components

    alpha: float (0,1)
        Confidence limit


    Returns
    -------
    limit: int
        Q-Residual limit corresponding to alpha

    """

    th1 = np.sum(eig_residuals)
    th2 = np.square(eig_residuals).sum()
    th3 = (eig_residuals ** 3).sum()
    h0 = 1 - (2 * th1 * th3) / (3 * th2 ** 2)
    limit = th1 * ((scipy.stats.norm.ppf(1 - alpha) * np.sqrt(2 * th2 * h0 ** 2)) / th1 + (th2 * h0 * (h0 - 1)) / (
            th1 ** 2) + 1) ** (1 / h0)
    return limit


def calculate_Q(X, loadings_res):
    """
    Calculate sum of squared residuals

    :param X:
    :param loadings_res:
    :return:
    """
    Q = X @ loadings_res @ loadings_res.T @ X.T
    return Q


def calculate_Q_incomplete(X, X_hat):
    """
    Calculate sum of squared residuals when the batch is incomplete

    Parameters
    ----------
    X: numpy array

    X_hat: numpy array

    Returns
    -------
    Q: float
        Sum of squared residuals
    """

    # Calculate residuals
    E = X - X_hat

    # Calculate Q
    Q = np.sum(E ** 2)

    return Q


def calculate_Q_fed(residuals_masked):
    """
    Calculate sum of squared residuals

    Parameters
    ----------
    residuals_masked

    Returns
    -------
    Q

    """
    residuals_masked_sum = np.sum(residuals_masked, axis=0)
    Q_masked = residuals_masked_sum @ residuals_masked_sum.T
    return Q_masked


def calculate_Q_variable_contributions(e):
    return e ** 2


def aggregate_spe(spe_arr):
    """
    Aggregate SPE calculated by data holders

    Parameters
    ----------
    spe_arr: list
        A list of SPE calculated by data holders

    Returns
    -------
    agg_spe: float
        Aggregated SPE
    """

    agg_spe = np.sum(spe_arr)

    return agg_spe


def calculate_spe(X, V, n_comp):
    """
    Calculate SPE

    Parameters
    ----------
    X: numpy array of shape (m, n)
        Data to be transformed

    V: numpy array of shape (n, n)
        Loading matrix

    n_comp: int
        Number of selected principal components

    Returns
    -------
    spe: float
        Squared prediction error


    """
    T = X @ V[:, :n_comp]
    X_hat = T @ V[:, :n_comp].T
    spe = np.sum((X - X_hat) ** 2, axis=1)
    return spe


def calculate_spe_batch(X, V, n_comp, J):
    """
    Calculate SPE for a batch

    Parameters
    ----------
    X: numpy array of shape (I, KJ)
        Data to be transformed with I, J, and K stand for the no. batches, no. variables, and no. time intervals
        respectively

    V: numpy array of shape (n, n)
        Loading matrix

    n_comp: int
        Number of selected principal components

    J: int
        Number of variables

    Returns
    -------
    spe: numpy array of shape (I, K)
        Squared prediction error at each time interval

    """
    T = X @ V[:, :n_comp]
    X_hat = T @ V[:, :n_comp].T

    # Calculate squared residuals
    E_squared = (X - X_hat) ** 2

    # Calculate no. batches and no. time intervals
    I = X.shape[0]  # No. batches
    K = int(X.shape[1] / J)  # No. time intervals

    # Reshape E_squared
    E_squared_reshaped = np.reshape(E_squared, (I, J, K), order='F')

    # Calculate spe
    spe = np.sum(E_squared_reshaped, axis=1)

    return spe


def plot_control_chart(scores, limit, title, x_label, y_label):
    """
    Plot Hotelling's T^2 control chart

    Parameters
    ----------
    scores: numpy array
        Scores

    limit:
        Confidence limit

    title

    x_label: str

    y_label: str

    Returns
    -------
        Plot

    """
    plt.scatter(range(1, len(scores) + 1), scores)
    plt.axhline(y=limit, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


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


