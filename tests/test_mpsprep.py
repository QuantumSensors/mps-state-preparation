"""
Unit tests for the MPS state preparation technique.
"""
import pickle
import numpy as np
import pytest


def test_LocalMPSTensor():
    from mpsprep import LocalMPSTensor

    LMT = LocalMPSTensor(left_bond_dim=2, right_bond_dim=2, physical_dim=4)

    assert LMT.left_bond_dim == 2
    assert LMT.right_bond_dim == 2
    assert LMT.physical_dim == 4

    for i in range(4):
        assert np.all(LMT[i] == 0.)

    LMT[0] = [[0, 1], [2, 3]]

    assert np.all(LMT[0] == [[0, 1], [2, 3]])

    for i in range(1, 4):
        assert np.all(LMT[i] == 0.)


def test_MatrixProductState():
    from mpsprep import MatrixProductState

    # num_qubits = 5, sparsity = 0.5, seed = 1
    y_tensor = np.array(
        [
            [
                [
                    [[0.42189239, 0.0], [0.07048027, 0.05880596]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                [
                    [[0.0, 0.43026305], [0.0, 0.0]],
                    [[0.22640647, 0.12301922], [0.21286003, 0.23744375]],
                ],
            ],
            [
                [
                    [[0.0, 0.0], [0.42551977, 0.08925864]],
                    [[0.0, 0.0], [0.12147602, 0.0]],
                ],
                [
                    [[0.17685127, 0.32919542], [0.31797543, 0.0]],
                    [[0.11508051, 0.05083186], [0.0, 0.0]],
                ],
            ],
        ]
    )

    MPS_cores, s_vals = MatrixProductState.svd_decompose(y_tensor)

    assert str(MPS_cores[0]) == "MPSLocalTensor A_(1,2,2)"
    assert str(MPS_cores[1]) == "MPSLocalTensor A_(2,2,4)"
    assert str(MPS_cores[2]) == "MPSLocalTensor A_(4,2,4)"
    assert str(MPS_cores[3]) == "MPSLocalTensor A_(4,2,2)"
    assert str(MPS_cores[4]) == "MPSLocalTensor A_(2,2,1)"

    s_vals_correct = [
        np.array([0.84433934, 0.5358088]),
        np.array([0.73258282, 0.51886774, 0.39687527, 0.19128175]),
        np.array([0.76265058, 0.46552832, 0.41302483, 0.17623269]),
        np.array([0.84894209, 0.52848589]),
        np.array([1.0]),
    ]

    for i in range(len(s_vals_correct)):
        assert np.allclose(s_vals[i], s_vals_correct[i],
                           atol=1e-10, rtol=1e-5)

    MPS_cores_correct = [
        [
            1,
            2,
            2,
            np.array([[-0.77010544, -0.63791662], [-0.63791662, 0.77010544]]),
            np.array([[-0.77010544, -0.63791662, -0.63791662, 0.77010544]]),
        ],
        [
            2,
            4,
            2,
            np.array(
                [
                    [-0.3763368, 0.7099092, -0.16902466, 0.57081539],
                    [-0.91517496, -0.35210517, -0.06555116, -0.18487777],
                    [0.12260633, -0.07327149, -0.98256833, -0.11898927],
                    [0.07611242, -0.60554287, -0.04114823, 0.79109516],
                ]
            ),
            np.array(
                [
                    [
                        -0.3763368,
                        -0.91517496,
                        0.7099092,
                        -0.35210517,
                        -0.16902466,
                        -0.06555116,
                        0.57081539,
                        -0.18487777,
                    ],
                    [
                        0.12260633,
                        0.07611242,
                        -0.07327149,
                        -0.60554287,
                        -0.98256833,
                        -0.04114823,
                        -0.11898927,
                        0.79109516,
                    ],
                ]
            ),
        ],
        [
            4,
            4,
            2,
            np.array(
                [
                    [-0.79108696, -0.48431316, 0.05822757, -0.26066956],
                    [-0.42406486, 0.1673445, -0.14730365, 0.39274967],
                    [0.33600521, -0.81990341, -0.13673803, 0.39570405],
                    [-0.23900847, 0.10009221, -0.11533235, 0.62552689],
                    [-0.04029039, 0.07141466, -0.93708028, -0.17792104],
                    [0.02996327, -0.0311508, -0.1366187, 0.15672951],
                    [-0.03181291, -0.19510734, 0.03728723, -0.28384822],
                    [0.14419669, -0.10506774, -0.21151332, -0.30519443],
                ]
            ),
            np.array(
                [
                    [
                        -0.79108696,
                        -0.42406486,
                        -0.48431316,
                        0.1673445,
                        0.05822757,
                        -0.14730365,
                        -0.26066956,
                        0.39274967,
                    ],
                    [
                        0.33600521,
                        -0.23900847,
                        -0.81990341,
                        0.10009221,
                        -0.13673803,
                        -0.11533235,
                        0.39570405,
                        0.62552689,
                    ],
                    [
                        -0.04029039,
                        0.02996327,
                        0.07141466,
                        -0.0311508,
                        -0.93708028,
                        -0.1366187,
                        -0.17792104,
                        0.15672951,
                    ],
                    [
                        -0.03181291,
                        0.14419669,
                        -0.19510734,
                        -0.10506774,
                        0.03728723,
                        -0.21151332,
                        -0.28384822,
                        -0.30519443,
                    ],
                ]
            ),
        ],
        [
            4,
            2,
            2,
            np.array(
                [
                    [-0.61354616, 0.3100792],
                    [-0.61837392, -0.16812243],
                    [-0.02728169, 0.82201987],
                    [0.19238042, -0.05287899],
                    [-0.3465109, -0.18380408],
                    [0.28179862, 0.24951046],
                    [-0.03203417, -0.05547546],
                    [0.05397991, -0.31297915],
                ]
            ),
            np.array(
                [
                    [-0.61354616, -0.61837392, 0.3100792, -0.16812243],
                    [-0.02728169, 0.19238042, 0.82201987, -0.05287899],
                    [-0.3465109, 0.28179862, -0.18380408, 0.24951046],
                    [-0.03203417, 0.05397991, -0.05547546, -0.31297915],
                ]
            ),
        ],
        [
            2,
            1,
            2,
            np.array([[0.74691612], [0.40350834], [0.25119318], [-0.46497238]]),
            np.array([[0.74691612, 0.40350834], [0.25119318, -0.46497238]]),
        ],
    ]

    for i, MPS_core in enumerate(MPS_cores_correct):
        assert MPS_cores[i].left_bond_dim == MPS_core[0]
        assert MPS_cores[i].right_bond_dim == MPS_core[1]
        assert MPS_cores[i].physical_dim == MPS_core[2]
        assert np.allclose(MPS_cores[i].to_left_matrix(), MPS_core[3],
                           atol=1e-10, rtol=1e-5)
        assert np.allclose(MPS_cores[i].to_right_matrix(), MPS_core[4],
                           atol=1e-10, rtol=1e-5)

    MPS_exact = MatrixProductState(MPS_cores)
    assert MPS_exact.num_qubits == 5
    assert np.all(
        MPS_exact.computational_states
        == np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
            ]
        )
    )

    MPS_exact.right_normalize("F")
    y_amp_true = np.array(
        [
            4.21892385e-01 + 0.0j,
            9.36588586e-16 + 0.0j,
            -3.02535774e-15 + 0.0j,
            1.76851274e-01 + 0.0j,
            -3.04351480e-17 + 0.0j,
            2.30603421e-16 + 0.0j,
            2.26406466e-01 + 0.0j,
            1.15080511e-01 + 0.0j,
            7.04802705e-02 + 0.0j,
            4.25519768e-01 + 0.0j,
            2.32256023e-15 + 0.0j,
            3.17975427e-01 + 0.0j,
            -8.70788522e-17 + 0.0j,
            1.21476022e-01 + 0.0j,
            2.12860028e-01 + 0.0j,
            -1.28432389e-16 + 0.0j,
            1.94289029e-16 + 0.0j,
            3.27723080e-17 + 0.0j,
            4.30263049e-01 + 0.0j,
            3.29195418e-01 + 0.0j,
            1.07639731e-19 + 0.0j,
            2.20520838e-17 + 0.0j,
            1.23019222e-01 + 0.0j,
            5.08318556e-02 + 0.0j,
            5.88059568e-02 + 0.0j,
            8.92586437e-02 + 0.0j,
            -7.15290427e-18 + 0.0j,
            -1.11022302e-16 + 0.0j,
            -7.79928989e-17 + 0.0j,
            -1.04083409e-16 + 0.0j,
            2.37443749e-01 + 0.0j,
            -1.60080640e-16 + 0.0j,
        ]
    )
    y_amp_test = MPS_exact.get_all_amplitudes()
    assert np.allclose(y_amp_test, y_amp_true, atol=1e-10, rtol=1e-5)

    assert np.isclose(MPS_exact.get_amplitude("00000"), 0.42189238528717427,
                      atol=1e-10, rtol=1e-5)
    with pytest.raises(TypeError):
        MPS_exact.get_amplitude(["00000"])

    MPS_trunc_cores_correct = [
        [
            1,
            2,
            2,
            np.array([[-0.77010544, -0.63791662], [-0.63791662, 0.77010544]]),
            np.array([[-0.77010544, -0.63791662, -0.63791662, 0.77010544]]),
        ],
        [
            2,
            3,
            2,
            np.array(
                [
                    [-0.3763368, 0.7099092, -0.16902466],
                    [-0.91517496, -0.35210517, -0.06555116],
                    [0.12260633, -0.07327149, -0.98256833],
                    [0.07611242, -0.60554287, -0.04114823],
                ]
            ),
            np.array(
                [
                    [
                        -0.3763368,
                        -0.91517496,
                        0.7099092,
                        -0.35210517,
                        -0.16902466,
                        -0.06555116,
                    ],
                    [
                        0.12260633,
                        0.07611242,
                        -0.07327149,
                        -0.60554287,
                        -0.98256833,
                        -0.04114823,
                    ],
                ]
            ),
        ],
        [
            3,
            3,
            2,
            np.array(
                [
                    [-0.80157934, 0.48562291, -0.0653271],
                    [-0.42882643, -0.17823133, 0.12443594],
                    [0.33421819, 0.83923852, 0.2198306],
                    [-0.24130484, -0.10220629, 0.10852754],
                    [-0.05320627, -0.13011223, 0.94814326],
                    [0.02859597, 0.02659644, 0.14549471],
                ]
            ),
            np.array(
                [
                    [
                        -0.80157934,
                        -0.42882643,
                        0.48562291,
                        -0.17823133,
                        -0.0653271,
                        0.12443594,
                    ],
                    [
                        0.33421819,
                        -0.24130484,
                        0.83923852,
                        -0.10220629,
                        0.2198306,
                        0.10852754,
                    ],
                    [
                        -0.05320627,
                        0.02859597,
                        -0.13011223,
                        0.02659644,
                        0.94814326,
                        0.14549471,
                    ],
                ]
            ),
        ],
        [
            3,
            2,
            2,
            np.array(
                [
                    [-0.63124842, 0.35440103],
                    [-0.60884794, -0.1528241],
                    [-0.01612181, -0.87106188],
                    [-0.16732155, 0.04675583],
                    [0.33690339, 0.12948441],
                    [-0.29844481, -0.2708202],
                ]
            ),
            np.array(
                [
                    [-0.63124842, -0.60884794, 0.35440103, -0.1528241],
                    [-0.01612181, -0.16732155, -0.87106188, 0.04675583],
                    [0.33690339, -0.29844481, 0.12948441, -0.2708202],
                ]
            ),
        ],
        [
            2,
            1,
            2,
            np.array([[0.76579079], [0.38949222], [0.23198735], [-0.45611638]]),
            np.array([[0.76579079, 0.38949222], [0.23198735, -0.45611638]]),
        ],
    ]

    MPS_cores_trunc, s_vals_trunc = MatrixProductState.svd_decompose(
        y_tensor, truncate_ranks=[2, 3, 3, 2, 1]
    )

    s_vals_trunc_correct = [
        np.array([0.84433934, 0.5358088]),
        np.array([0.73258282, 0.51886774, 0.39687527]),
        np.array([0.75449523, 0.4544367, 0.40301925]),
        np.array([0.83217961, 0.49565854]),
        np.array([0.9686074]),
    ]

    for i in range(len(s_vals_trunc_correct)):
        assert np.allclose(s_vals_trunc[i], s_vals_trunc_correct[i],
                           atol=1e-10, rtol=1e-5)

    for i, MPS_core in enumerate(MPS_trunc_cores_correct):
        assert MPS_cores_trunc[i].left_bond_dim == MPS_core[0]
        assert MPS_cores_trunc[i].right_bond_dim == MPS_core[1]
        assert MPS_cores_trunc[i].physical_dim == MPS_core[2]
        assert np.allclose(MPS_cores_trunc[i].to_left_matrix(), MPS_core[3],
                           atol=1e-10, rtol=1e-5)
        assert np.allclose(MPS_cores_trunc[i].to_right_matrix(), MPS_core[4],
                           atol=1e-10, rtol=1e-5)

    assert MatrixProductState.plot_singular_values(s_vals) is None

    MPS_cores_int_trunc, s_vals_int_trunc = MatrixProductState.svd_decompose(
        y_tensor, truncate_ranks=2
    )

    s_vals_int_trunc_correct = [
        np.array([0.84433934, 0.5358088]),
        np.array([0.73258282, 0.51886774]),
        np.array([0.75336658, 0.45317107]),
        np.array([0.75410641, 0.45193887]),
        np.array([0.87916166]),
    ]

    for i in range(len(s_vals_int_trunc_correct)):
        assert np.allclose(s_vals_int_trunc[i], s_vals_int_trunc_correct[i],
                           atol=1e-10, rtol=1e-5)


def test_MatrixProductInitializer():
    from mpsprep import MatrixProductInitializer, MatrixProductState

    # num_qubits = 5, sparsity = 0.5, seed = 1
    y_tensor = np.array(
        [
            [
                [
                    [[0.42189239, 0.0], [0.07048027, 0.05880596]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                [
                    [[0.0, 0.43026305], [0.0, 0.0]],
                    [[0.22640647, 0.12301922], [0.21286003, 0.23744375]],
                ],
            ],
            [
                [
                    [[0.0, 0.0], [0.42551977, 0.08925864]],
                    [[0.0, 0.0], [0.12147602, 0.0]],
                ],
                [
                    [[0.17685127, 0.32919542], [0.31797543, 0.0]],
                    [[0.11508051, 0.05083186], [0.0, 0.0]],
                ],
            ],
        ]
    )

    MPS_cores, _ = MatrixProductState.svd_decompose(y_tensor)

    MPS = MatrixProductState(MPS_cores)
    MPS.right_normalize("F")
    MPI = MatrixProductInitializer(MPS)
    assert MPI.num_qubits == 5

    unitaries = MPI.gate_unitaries(False)

    for unitary in unitaries:
        assert MatrixProductInitializer.is_unitary(unitary)

    with open("tests/unitaries.pkl", "rb") as handle:
        unitaries_correct = pickle.load(handle)
    for i in range(len(unitaries_correct)):
        assert np.allclose(unitaries[i], unitaries_correct[i],
                           atol=1e-10, rtol=1e-5)

    with open("tests/unitaries_full.pkl", "rb") as handle:
        unitaries_full_correct = pickle.load(handle)
    unitaries_full = MPI.gate_unitaries(True)

    for i in range(len(unitaries_full_correct)):
        u_full_dense = unitaries_full[i].todense()
        assert np.allclose(u_full_dense,
                           unitaries_full_correct[i].todense(),
                           atol=1e-10, rtol=1e-5)
        assert MatrixProductInitializer.is_unitary(u_full_dense)

    with open("tests/gate_unitary.pkl", "rb") as handle:
        gate_unitary_correct = pickle.load(handle)
    assert np.allclose(MPI.gate_unitary.todense(),
                       gate_unitary_correct.todense(), atol=1e-7, rtol=1e-5)
