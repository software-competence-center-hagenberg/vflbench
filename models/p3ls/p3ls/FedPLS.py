# Import libraries
# from statsmodels.multivariate.pca import PCA
import numpy as np
from sklearn.utils.extmath import svd_flip
from fedpls.utils import _svd_flip_1d


class FedPLS:
    """
    Federated Partial Least Square

    """

    def __init__(self, n_comp=None, method="svd"):
        self.type = "Federated PLS"  # Type of model
        self.max_n_comp = None
        self.n_comp = n_comp  # No. components to keep
        self.method = method  # Method for calculating principal components
        self.x_weights_masked = None
        self.y_weights_masked = None
        self.x_loadings_masked = None
        self.y_loadings_masked = None
        self.x_rotations_masked = None
        self.y_rotations_masked = None
        self.x_scores_masked = None
        self.coef_masked = None
        self.g_t_masked = None

    def fit(self, X_masked, Y_masked): # X_i_masked_list
        """Fit model to data.
        Parameters
        ----------
        # X_i_masked_list : array-like of shape (n_samples, n_features)
        #     Training vectors, where `n_samples` is the number of samples and
        #     `n_features` is the number of predictors.
        X_masked : array-like of shape (n_samples, n_features)
        Y_masked : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.
        Returns
        -------
        self : object
            Fitted model.
        """

        # # Perform secure aggregation
        # X_masked = np.sum(X_i_masked_list, axis=0)

        # Define the maximum no. principal components
        self.max_n_comp = min(X_masked.shape)

        if self.n_comp is None:
            self.n_comp = self.max_n_comp

        elif (self.n_comp < 0) | (self.n_comp > self.max_n_comp):
            ArithmeticError('No. principal components have to be positive and smaller than max_n_comp '
                            'that is defined by the data matrix.'
                            )

        # Perform SVD
        m = X_masked.shape[0]
        n = X_masked.shape[1]
        l = Y_masked.shape[1]

        W = np.zeros((n, self.n_comp))
        Z = np.zeros((l, self.n_comp))
        T = np.zeros((m, self.n_comp))
        C = np.zeros((l, self.n_comp))
        P = np.zeros((n, self.n_comp))

        E = X_masked.copy()
        F = Y_masked.copy()

        for i in range(self.n_comp):
            U, _, Vt = np.linalg.svd(E.T @ F, full_matrices=False)
            w = U[:, [0]]  # X's weights
            z = Vt[[0], :]  # y's weights

            _svd_flip_1d(w, z)

            t = E @ w  # X's scores
            p = t.T @ E / (t.T @ t)  # X's loadings
            c = t.T @ F / (t.T @ t)  # y's loadings

            W[:, i] = w.reshape(W[:, i].shape)
            Z[:, i] = z.reshape(Z[:, i].shape)
            T[:, i] = t.reshape(T[:, i].shape)
            P[:, i] = p.reshape(P[:, i].shape)
            C[:, i] = c.reshape(C[:, i].shape)

            E = E - t @ p
            F = F - t @ c

        # Get scores, loadings, and other properties
        self.x_weights_masked = W
        self.y_weights_masked = Z
        self.x_loadings_masked = P
        self.y_loadings_masked = C
        self.x_rotations_masked = W @ np.linalg.pinv(P.T @ W)
        self.y_rotations_masked = Z @ np.linalg.pinv(C.T @ Z)
        self.x_scores_masked = T
        self.coef_masked = W @ np.linalg.pinv(P.T @ W) @ C.T

    def transform(self, X_masked):  # X_masked_list
        # Perform secure aggregation
        # X_masked = np.sum(X_masked_list, axis=0)
        X_scores_masked = X_masked @ self.x_rotations_masked

        return X_scores_masked

    def get_x_weights_masked(self, H_masked):
        """
        Compute the masked x_weights for data holder i

        Parameters
        ----------
        H_masked: numpy array of shape (,)
            The key to recover the masked loading matrix

        Returns
        -------
        x_weights_masked: numpy array of shape (,)
            The masked loading matrix

        """

        x_weights_masked = H_masked @ self.x_weights_masked

        return x_weights_masked

    def get_y_weights_masked(self):
        """
        Compute the masked y_weights for data holder i

        Parameters
        ----------

        Returns
        -------
        y_weights_masked: numpy array of shape (,)
            The masked loading matrix

        """

        return self.y_weights_masked

    def get_x_loadings_masked(self, H_masked):
        """
        Compute the masked x_loadings for data holder i

        Parameters
        ----------
        H_masked: numpy array of shape (,)
            The key to recover the masked loading matrix

        Returns
        -------
        x_loadings_masked: numpy array of shape (,)
            The masked loading matrix

        """

        x_loadings_masked = H_masked @ self.x_loadings_masked

        return x_loadings_masked

    def get_y_loadings_masked(self):
        """
        Compute the masked y_loadings for data holder i

        Parameters
        ----------

        Returns
        -------
        y_loadings_masked: numpy array of shape (,)
            The masked loading matrix

        """

        return self.y_loadings_masked

    def get_x_rotations_masked(self, H_masked):
        """
        Compute the masked x_loadings for data holder i

        Parameters
        ----------
        H_masked: numpy array of shape (,)
            The key to recover the masked rotations matrix

        Returns
        -------
        x_rotations_masked: numpy array of shape (,)
            The masked rotations matrix

        """

        x_rotations_masked = H_masked @ self.x_rotations_masked

        return x_rotations_masked

    def get_y_rotations_masked(self):
        """
        Compute the masked y_rotations for data holder i

        Parameters
        ----------

        Returns
        -------
        y_rotations_masked: numpy array of shape (,)
            The masked loading matrix

        """

        return self.y_rotations_masked

    def get_x_scores_masked(self):
        """
        Compute the masked x_loadings for data holder i

        Parameters
        ----------

        Returns
        -------
        x_scores_masked: numpy array of shape (,)
            The masked scores matrix

        """

        return self.x_scores_masked

    def get_coef_masked(self, H_masked):
        """
        Compute the masked coef for data holder i

        Parameters
        ----------

        Returns
        -------
        coef_masked: numpy array of shape (,)
            The masked coefficients matrix

        """

        coef_masked = H_masked @ self.coef_masked @ self.g_t_masked

        return coef_masked

    def set_g_t_masked(self, G_T_masked):
        self.g_t_masked = G_T_masked

    @staticmethod
    def predict(local_preds):
        """
        Compute the masked coef for data holder i

        Parameters
        ----------

        Returns
        -------
        preds: numpy array of shape (,)
            The masked predictions

        """

        preds = np.sum(local_preds, axis=0)

        return preds
