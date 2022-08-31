"""
A way to efficiently encode amplitudes into quantum registers with the ability
to optimally approximate the target state by finding similar states that have
lower entanglement.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import rq
from scipy.linalg import null_space
import scipy.sparse
from mpsprep.helpers import bit_string, update_kwargs_dict


class LocalMPSTensor:
    """
    Cores of the tensor network. These are the tensors that are multiplied
    together to form the Matrix Product State (MPS).
    """
    def __init__(self, tensor=None, left_bond_dim=None, right_bond_dim=None,
                 physical_dim=None):
        """
        Initialize the tensor.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor.
        left_bond_dim : int
            The left bond dimension.
        right_bond_dim : int
            The right bond dimension.
        physical_dim : int
            The physical dimension.
        """
        if tensor is not None:
            assert len(tensor.shape) == 3
            self._A = tensor
        else:
            self._A = np.zeros((left_bond_dim, physical_dim, right_bond_dim))

    @property
    def left_bond_dim(self):
        """
        The dimension of the left bond.

        :type: int
        """
        return self._A.shape[0]

    @property
    def right_bond_dim(self):
        """
        The dimension of the right bond.

        :type: int
        """
        return self._A.shape[-1]

    @property
    def physical_dim(self):
        """
        The physical dimension of the tensor.

        :type: int
        """
        return self._A.shape[1]

    def __getitem__(self, key):
        return self._A[:, key, :]

    def __setitem__(self, key, value):
        self._A[:, key, :] = value

    def __repr__(self):
        out = "MPSLocalTensor "
        A = (f"A_({self.left_bond_dim},{self.physical_dim},"
             f"{self.right_bond_dim})")
        out += A
        return out

    def to_left_matrix(self):
        """
        Convert to a left matrix.

        Returns
        -------
        matrix : np.ndarray
            The left matrix.
        """
        return np.reshape(self._A, (self.left_bond_dim*self.physical_dim,
                                    self.right_bond_dim))

    def to_right_matrix(self, order="F"):
        """
        Convert to a right matrix.

        Parameters
        ----------
        order : {"F", "C"}, optional
            The layout of the matrix in memory. Default is "F".

        Returns
        -------
        matrix : np.ndarray
            The right matrix.
        """
        return np.reshape(self._A, (self.left_bond_dim,
                                    self.physical_dim*self.right_bond_dim),
                          order)


class MatrixProductState:
    """
    A class for representing a matrix product state.
    """

    def __init__(self, tensor_list):
        """
        Initializes a Matrix Product State. Given a list of
        :py:class:`LocalMPSTensor` representing
        :math:`[G[j_0], G[j_1], ..., G[j_N-1]]`,
        this initializes the matrix product state
        :math:`|MPS> = \\sum_{j_0 ... j_N-1} G[j_0] G[j_1] ... G[j_N-1]
        x |j_N-1, ...,j_0 >`.

        Note that :math:`G[j_0] G[j_1] ... G[j_N-1]` are matrix products, and
        the order is therefore important.

        Parameters
        ----------
        tensor_list : list of :py:class:`LocalMPSTensor`
            List of tensors that describes the desired MPS state.
        """
        self._tensors = tensor_list

    @property
    def num_qubits(self):
        """
        The number of qubits in the MPS.

        :type: int
        """
        return len(self._tensors)

    @property
    def computational_states(self):
        """
        The computational basis states of the MPS.

        :type: numpy.ndarray
        """
        return np.arange(2**self.num_qubits)

    def get_amplitude(self, state):
        """
        Return the amplitude of the state in the MPS.

        Parameters
        ----------
        state : int
            The computational basis state of the MPS.

        Returns
        -------
        state_amplitude : float
            The amplitude of the state in the MPS.
        """
        num_qubits = self.num_qubits
        if isinstance(state, (int, np.integer)):
            bits = bit_string(num_qubits, state)
        elif isinstance(state, str):
            bits = state
        else:
            raise TypeError(f"state is type {type(state)} but should be"
                            " either int or str.")

        tensors = self._tensors
        # Assume that state and bitstring uses a little-endian format, and that
        # tensors[0] corresponds to the first qubit
        matrix_list = [tensors[m][int(bits[-(m + 1)])]
                       for m in range(num_qubits)]
        out = self.contract_matrices(matrix_list)
        return out

    def get_all_amplitudes(self):
        """
        Return the amplitudes of all states in the MPS.

        Returns
        -------
        all_amplitudes : numpy.ndarray
            The amplitudes of all states in the MPS.
        """
        states = self.computational_states
        amp = np.zeros(states.shape, dtype=np.complex128)
        for m, state in enumerate(states):
            amp[m] = self.get_amplitude(state)
        return amp

    @staticmethod
    def contract_matrices(matrix_list):
        """
        Contract a list of matrices.

        Parameters
        ----------
        matrix_list : list of np.ndarray
            List of matrices to be contracted.

        Returns
        -------
        matrix : np.ndarray
            The contracted matrix.
        """
        if len(matrix_list) == 1:
            out = np.sum(matrix_list[0])
        else:
            A_now = matrix_list[0]
            for m in range(1, len(matrix_list)):
                A_now = np.matmul(A_now, matrix_list[m])
            out = np.sum(A_now)
        return out

    def right_normalize(self, order="F"):
        """
        Right normalize the MPS.

        Parameters
        ----------
        order : {"F", "C"}, optional
            The layout of the matrix in memory. Default is "F".

        Returns
        -------
        None
        """
        if self.num_qubits > 1:
            tensor = self._tensors[-1]
            R, Q = rq(tensor.to_right_matrix(order), mode="economic")
            tensor._A = np.reshape(Q, tensor._A.shape, order)
            for m in range(2, self.num_qubits + 1):
                tensor = self._tensors[-m]
                tensor._A = np.einsum("ijk,kl->ijl", tensor._A, R)
                if m < self.num_qubits:
                    R, Q = rq(tensor.to_right_matrix(order), mode="economic")
                    tensor._A = np.reshape(Q, tensor._A.shape, order)

    @staticmethod
    def svd_decompose(A, truncate_ranks=None):
        """
        Decomposes the tensor A into a list of :py:class:`LocalMPSTensor` using
        the TT-SVD algorithm. Typically, A gives the expansion coefficients of
        an MPS state as follows:
        :math:`|MPS> = \\sum_{j_0 ... j_N-1} A(j_0, ..., j_N-1)
        |j_N-1, ... ,j_0>`.

        Note that :math:`j_0` is assumed to be in the first dimension/axis of
        A.

        Parameters
        ----------
        A : np.ndarray of shape (2**N) for integer N
            The tensor to decompose.
        truncate_ranks : list of int, optional
            The number of singular values to preserve at each SVD step. This
            defines the aggressiveness of the approximation scheme. If None,
            all singular values are preserved. Default is None.

        Returns
        -------
        MPS_tensors : List of :py:class:`LocalMPSTensor`
            Desired MPS tensors.
        singular_values : List of np.ndarray
            Returns the singular values obtained at each stage of the
            decomposition.
        """
        dims = len(A.shape)
        tensor_cores = []
        singular_values = []
        if truncate_ranks is not None:
            if isinstance(truncate_ranks, (int, np.integer)):
                truncate_ranks = [truncate_ranks]*dims

        # Perform first decomposition
        A_matrix = A.reshape((A.shape[0], np.prod(A.shape[1:])))
        u1, s1, vT = np.linalg.svd(A_matrix, False)
        if truncate_ranks is not None:
            u1 = u1[..., :truncate_ranks[0]]
            s1 = s1[:truncate_ranks[0]]
            vT = vT[:truncate_ranks[0], ...]
        tensor_cores.append(u1.reshape((1, A.shape[0], -1)))
        singular_values.append(s1)
        B = np.matmul(np.diag(s1), vT)

        # Do the rest
        for m in range(1, dims):

            B_matrix = B.reshape((B.shape[0]*A.shape[m], -1))
            u1, s1, vT = np.linalg.svd(B_matrix, False)
            if truncate_ranks is not None:
                u1 = u1[..., :truncate_ranks[m]]
                s1 = s1[:truncate_ranks[m]]
                vT = vT[:truncate_ranks[m], ...]
            tensor_cores.append(u1.reshape((B.shape[0], A.shape[m], -1)))
            singular_values.append(s1)
            B = np.matmul(np.diag(s1), vT)

        MPS_tensors = [LocalMPSTensor(tensor) for tensor in tensor_cores]

        return MPS_tensors, singular_values

    @staticmethod
    def plot_singular_values(s_vals, subplot_kwargs=None, yscale="log",
                             ylabel="Singular values", bar_kwargs=None):
        """
        Plot the singular values of an MPS. Does not explicitly return a
        Matplotlib figure or plot object.

        Parameters
        ----------
        s_vals : list of np.ndarray
            The singular values of the MPS SVD.
        subplot_kwargs : dict, optional
            Keyword arguments for the subplot. Default is None.
        yscale : str, optional
            The scale of the y-axis. Either "log" or "linear". Default is
            "log".
        ylabel : str, optional
            The label of the y-axis. Default is "Singular values".
        bar_kwargs : dict, optional
            Keyword arguments for the bar plot. Default is None.

        Returns
        -------
        None
        """
        max_length = np.max([len(s_val) for s_val in s_vals])
        max_sval = np.max([np.max(s_val) for s_val in s_vals])
        min_sval = np.min([np.min(s_val) for s_val in s_vals])

        min_sval = max(min_sval, 1e-20)

        subplot_kwargs0 = {"ncols": len(s_vals),
                           "figsize": (2.5*len(s_vals), 4.5),
                           "sharey": True}
        subplot_kwargs = update_kwargs_dict(subplot_kwargs0, subplot_kwargs)
        bar_kwargs0 = {"width": 1, "edgecolor": "C1"}
        bar_kwargs = update_kwargs_dict(bar_kwargs0, bar_kwargs)

        _, axes = plt.subplots(**subplot_kwargs)
        for m, ax in enumerate(axes):
            s_val = s_vals[m]
            if m == 0:
                ax.set_yscale(yscale)
                ax.set_ylabel(ylabel)
            ax.bar(np.arange(len(s_val)), s_val, **bar_kwargs)
            ax.set_xlim(-0.5, max_length - 0.5)
            ax.set_ylim(min_sval/2, max_sval)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel(f"Core {m+1}")
        plt.show()


class MatrixProductInitializer:
    """
    Generate the unitaries needed to prepare the desired state given the
    desired Matrix Product State (MPS).
    """

    def __init__(self, MPS):
        """
        Initializes the MatrixProductInitializer (MPI) class.

        Parameters
        ----------
        MPS : :py:class:`MatrixProductState`
            The MPS to initialize.
        """
        self.MPS = MPS

    @staticmethod
    def _gate_unitary(core):
        """
        Returns the unitary matrix of a single gate.

        Parameters
        ----------
        core : np.ndarray of shape (2, 2, 2, 2)
            The core of the gate.

        Returns
        -------
        gate_unitary : np.ndarray of shape (2, 2, 2, 2)
            The unitary matrix of the gate.
        """
        N = int(np.ceil(np.log2(core.left_bond_dim)))
        M = int(np.ceil(np.log2(core.right_bond_dim)))
        assert N <= (M + 1), "Bond dimensions are not right-normalizable."
        size = 2**(M + 1)
        G = np.zeros((size, size), dtype=np.complex128)
        rows = 2*core.right_bond_dim
        cols = core.left_bond_dim
        G[:rows, :cols] = core.to_right_matrix("F").T

        U_null = null_space(G[:, :cols].T)
        G[:, cols:] = U_null.conj()

        return G

    def _to_full_space(self, core_pos, gate_unitary):
        """
        Expands the gate_unitary in its native space into the full Hilbert
        space. core_pos is a zero based index of the MPS core that gate_unitary
        corresponds to.

        Parameters
        ----------
        core_pos : int
            The zero based index of the MPS core that gate_unitary corresponds
            to.
        gate_unitary : np.ndarray
            The gate unitary to expand.

        Returns
        -------
        gate_unitary : np.ndarray
            The expanded gate unitary.
        """
        num_qubits = self.num_qubits
        pre_qubits = core_pos
        gate_qubits = int(np.log2(gate_unitary.shape[-1]))
        post_qubits = num_qubits - pre_qubits - gate_qubits
        U = gate_unitary

        if pre_qubits > 0:
            U = scipy.sparse.kron(U, scipy.sparse.eye(int(2**pre_qubits)))
        if post_qubits > 0:
            U = scipy.sparse.kron(scipy.sparse.eye(int(2**post_qubits)), U)

        return U

    @property
    def num_qubits(self):
        """
        The number of qubits in the MPS.

        :type: int
        """
        return self.MPS.num_qubits

    @staticmethod
    def is_unitary(matrix, rtol=1e-5, atol=1e-8, equal_nan=False):
        """
        Returns True if the matrix is a unitary matrix within the specified
        tolerance.

        Parameters
        ----------
        matrix : np.ndarray
            The matrix to check.
        rtol : float, optional
            The relative tolerance. Default is 1e-5.
        atol : float, optional
            The absolute tolerance. Default is 1e-8.
        equal_nan : bool, optional
            Whether to treat NaN as equal. Default is False.

        Returns
        -------
        is_unitary : bool
            Whether the matrix is a unitary matrix.
        """
        out = np.allclose(np.matmul(np.conj(matrix.T), matrix),
                          np.eye(matrix.shape[-1]), rtol, atol, equal_nan)
        return out

    def gate_unitaries(self, full_space=True):
        """
        Returns a list of unitaries :math:`[U_1, U_2, ..., U_N]` such that
        :math:`|MPS> = U_N ... U_2 U_1 |0>`.

        Parameters
        ----------
        full_space : bool, optional
            If True, all of the unitaries are returned in the full Hilbert
            space of the MPS. Otherwise, they are returned in their native
            space and will either be 4x4 matrices or 2x2 matrices.

        Returns
        -------
        gate_unitaries : list
            A list of unitaries.
        """

        MPS = self.MPS
        U_list = [self._gate_unitary(tensor) for tensor in MPS._tensors]

        if full_space:
            U_full_list = []
            for m, U in enumerate(U_list):
                U_full_list.append(self._to_full_space(m, U))
            out = U_full_list
        else:
            out = U_list

        return out

    @property
    def gate_unitary(self):
        """
        The full gate unitary of the MPS.

        :type: np.ndarray
        """
        U_full_list = self.gate_unitaries()

        U_full = U_full_list[0]
        for m in range(1, len(U_full_list)):
            U_full = U_full_list[m] * U_full

        return U_full
