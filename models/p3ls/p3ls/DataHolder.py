# Import libraries
from fedpls.utils import *


class DataHolder:
    """
    Data holder
    """

    def __init__(self, name, X, Y, A, H, G):
        self.name = name  # Data holder's name
        self.X = X  # Private data
        self.Y = Y  # Targets
        self.A = A  # The left key for masking X and Y
        self.H = H  # The right key for masking X
        self.G = G  # The right key for masking Y
        self.C = generate_rand_square_mat(X.shape[1])  # The key for masking Hi before sending it to the server
        # self.C = generate_orthogonal_matrix(X.shape[1])  # The key for masking Hi before sending it to the server
        self.C_inverse = np.linalg.pinv(self.C)  # self.C.T
        self.H_encrypted = self.C @ self.H  # Encrypted H_i
        self.X_encrypted = mask_data(X, A, H)  # Encrypted X

        if Y is not None:
            self.Y_encrypted = mask_data(Y, A, G)  # Encrypted Y

        self.x_weights = None  # Private X weights
        self.x_loadings = None  # Private X loadings
        self.x_rotations = None  # Private X rotations
        self.x_scores = None  # Private X scores
        self.y_weights = None  # Private Y weights
        self.y_loadings = None  # Private Y loadings
        self.y_rotations = None  # Private Y rotations
        self.y_scores = None  # Private Y scores
        self.coef = None  # Private regression coefficients

    def recover_x_weights(self, x_weights_masked):
        """
        Recover the X weights matrix for the ith data holder

        Parameters
        ----------
        x_weights_masked: numpy array of shape (ni, n_comp)
            The masked loading matrix computed by the server

        Returns
        -------
        x_weights: numpy array of shape (ni, n_comp)
            The recovered weights matrix

        """
        x_weights = self.C_inverse @ x_weights_masked
        self.x_weights = x_weights

        return x_weights

    def recover_x_loadings(self, x_loadings_masked):
        """
        Recover the X loadings matrix for the ith data holder

        Parameters
        ----------
        x_loadings_masked: numpy array of shape (ni, n_comp)
            The masked loading matrix computed by the server

        Returns
        -------
        x_loadings: numpy array of shape (ni, n_comp)
            The recovered loadings matrix

        """
        x_loadings = self.C_inverse @ x_loadings_masked
        self.x_loadings = x_loadings

        return x_loadings

    def recover_x_rotations(self, x_rotations_masked):
        """
        Recover the X rotations matrix for the ith data holder

        Parameters
        ----------
        x_rotations_masked: numpy array of shape (ni, n_comp)
            The masked rotation matrix computed by the server

        Returns
        -------
        x_rotations: numpy array of shape (ni, n_comp)
            The recovered rotations matrix

        """
        x_rotations = self.C_inverse @ x_rotations_masked
        self.x_rotations = x_rotations

        return x_rotations

    def recover_y_weights(self, y_weights_masked):
        """
        Recover the Y weights matrix for the ith data holder

        Parameters
        ----------
        y_weights_masked: numpy array of shape (ni, n_comp)
            The masked loading matrix computed by the server

        Returns
        -------
        x_weights: numpy array of shape (ni, n_comp)
            The recovered weights matrix

        """
        y_weights = self.G @ y_weights_masked
        self.y_weights = y_weights

        return y_weights

    def recover_y_loadings(self, y_loadings_masked):
        """
        Recover the Y loadings matrix for the ith data holder

        Parameters
        ----------
        y_loadings_masked: numpy array of shape (ni, n_comp)
            The masked loading matrix computed by the server

        Returns
        -------
        y_loadings: numpy array of shape (ni, n_comp)
            The recovered loadings matrix

        """
        y_loadings = self.G @ y_loadings_masked
        self.y_loadings = y_loadings

        return y_loadings

    def recover_y_rotations(self, y_rotations_masked):
        """
        Recover the X rotations matrix for the ith data holder

        Parameters
        ----------
        y_rotations_masked: numpy array of shape (ni, n_comp)
            The masked rotation matrix computed by the server

        Returns
        -------
        y_rotations: numpy array of shape (ni, n_comp)
            The recovered rotations matrix

        """
        y_rotations = self.G @ y_rotations_masked
        self.y_rotations = y_rotations

        return y_rotations

    def recover_coef(self, coef_masked, N):
        """
        Recover the coefficient matrix for the ith data holder

        Parameters
        ----------
        coef_masked: numpy array of shape (ni, n_comp)
            The masked coefficient matrix computed by the server

        N: numpy array of shape (l, l)
            An orthogonal matrix generated by the TA

        Returns
        -------
        coef: numpy array of shape (ni, l)
            The recovered rotations matrix

        """
        N_inverse = np.linalg.pinv(N)
        coef = self.C_inverse @ coef_masked @ N_inverse
        self.coef = coef

        return coef

    # def recover_coef(self, coef_masked):
    #     """
    #     Recover the coefficient matrix for the ith data holder
    #
    #     Parameters
    #     ----------
    #     coef_masked: numpy array of shape (ni, n_comp)
    #         The masked coefficient matrix computed by the server
    #
    #     Returns
    #     -------
    #     coef: numpy array of shape (ni, l)
    #         The recovered rotations matrix
    #
    #     """
    #     coef = np.linalg.pinv(self.C) @ coef_masked @ self.G.T
    #     self.coef = coef
    #
    #     return coef

    def predict(self, X, pred_type="targets"):

        if pred_type == "targets":
            pred = X @ self.coef
        else:  # "scores"
            pred = X @ self.x_rotations

        return pred

    # def recover_scores(self, T_masked):
    #     """
    #     Recover the score matrix
    #
    #     Parameters
    #     ----------
    #
    #     T_masked: numpy array of shape (m, n)
    #         The masked score matrix calculated by the federated server
    #
    #     Returns
    #     -------
    #     T: numpy array of shape (m, n)
    #         The recovered score matrix
    #
    #     """
    #     T = self.P.T @ T_masked
    #     return T
    #
    # def calculate_T2_contributions(self, scores, eigenvalues):
    #     """
    #     Calculate T2 contributions
    #
    #     Parameters
    #     ----------
    #     scores: numpy array of shape (1, n_comp)
    #         Scores of the sample after being projected on the model
    #
    #     eigenvalues: numpy array of shape (n_comp, )
    #         Eigenvalues of the model
    #
    #     Returns
    #     -------
    #     T2_cont: numpy array of shape (1, ni)
    #         T2 contributions
    #         ni is the number of variables that the participant have
    #
    #
    #     References
    #     ----------
    #
    #     https://wiki.eigenvector.com/index.php?title=T-Squared_Q_residuals_and_Contributions
    #
    #     """
    #     T2_cont = scores @ np.diag(np.sqrt(eigenvalues)) @ self.Vi[:, :len(scores)].T
    #     return T2_cont
    #
    # def calculate_Q(self, x, scores, n_comp, masking_key):
    #     """
    #     Calculate Q contributions
    #
    #     Parameters
    #     ----------
    #     x:
    #     scores:
    #     n_comp:
    #     masking_key:
    #
    #     Returns
    #     -------
    #
    #     """
    #     e = x - scores @ self.Vi[:, :n_comp].T
    #     Q_masked = e @ e.T * masking_key
    #
    #     return Q_masked
    #
    # def calculate_Q_contributions(self, X, scores, n_comp):
    #     """
    #     Calculate Q contributions
    #
    #     Parameters
    #     ----------
    #     scores:
    #     n_comp:
    #
    #     Returns
    #     -------
    #
    #     """
    #     e = X - scores @ self.Vi[:, :n_comp].T
    #     contrib = e**2
    #
    #     return contrib
