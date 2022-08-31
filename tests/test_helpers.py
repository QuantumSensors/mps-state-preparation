"""
Unit tests for the helper functions used in the MPS state preparation technique.
"""
import numpy as np
import pytest


def test_best_s_val_truncation_idx():
    from mpsprep.helpers import best_s_val_truncation_idx

    s_vals = [
        np.array([0.84433934, 0.5358088]),
        np.array([0.73258282, 0.51886774, 0.39687527, 0.19128175]),
        np.array([0.76265058, 0.46552832, 0.41302483, 0.17623269]),
        np.array([0.84894209, 0.52848589]),
        np.array([1.0]),
    ]

    assert np.all(
        best_s_val_truncation_idx(s_vals) == np.array([2.0, 3.0])
    )


def test_truncate_s_vals():
    from mpsprep.helpers import truncate_s_vals

    s_vals = [
        np.array([0.84433934, 0.5358088]),
        np.array([0.73258282, 0.51886774, 0.39687527, 0.19128175]),
        np.array([0.76265058, 0.46552832, 0.41302483, 0.17623269]),
        np.array([0.84894209, 0.52848589]),
        np.array([1.0]),
    ]

    truncated_s_vals_correct = [
        np.array([0.84433934, 0.5358088]),
        np.array([0.73258282, 0.51886774, 0.39687527, 0.19128175]),
        np.array([0.76265058, 0.46552832, 0.41302483]),
        np.array([0.84894209, 0.52848589]),
        np.array([1.0]),
    ]

    truncated_s_vals = truncate_s_vals(s_vals, np.array([2.0, 3.0]))
    for i in range(len(truncated_s_vals_correct)):
        assert np.allclose(truncated_s_vals[i], truncated_s_vals_correct[i],
                           atol=1e-10, rtol=1e-5)


def test_coarse_truncate_s_vals():
    from mpsprep.helpers import coarse_truncate_s_vals

    # num_qubits = 5, sparsity = 0.9, seed = 1

    s_vals_exact = [
        np.array([0.93022829, 0.36698137]),
        np.array([9.20686575e-01, 3.90302743e-01,
                  1.02846651e-16, 1.38511145e-17]),
        np.array([9.20686575e-01, 3.90302743e-01,
                  1.02846651e-16, 1.38777878e-17]),
        np.array([0.92068657, 0.39030274]),
        np.array([1.]),
    ]

    coarse_truncated_s_vals_correct = [
        np.array([0.93022829, 0.36698137]),
        np.array([0.92068657, 0.39030274]),
        np.array([0.92068657, 0.39030274]),
        np.array([0.92068657, 0.39030274]),
        np.array([1.0]),
    ]

    coarse_truncated_s_vals = coarse_truncate_s_vals(s_vals_exact)
    for i in range(len(coarse_truncated_s_vals_correct)):
        assert np.allclose(coarse_truncated_s_vals_correct[i],
                           coarse_truncated_s_vals[i], atol=1e-10, rtol=1e-5)


def test_mean_fractional_entropy():
    from mpsprep.helpers import mean_fractional_entropy

    y_amp = np.array(
        [
            0.42189239,
            0.0,
            0.0,
            0.17685127,
            0.0,
            0.0,
            0.22640647,
            0.11508051,
            0.07048027,
            0.42551977,
            0.0,
            0.31797543,
            0.0,
            0.12147602,
            0.21286003,
            0.0,
            0.0,
            0.0,
            0.43026305,
            0.32919542,
            0.0,
            0.0,
            0.12301922,
            0.05083186,
            0.05880596,
            0.08925864,
            0.0,
            0.0,
            0.0,
            0.0,
            0.23744375,
            0.0,
        ]
    )

    assert np.isclose(mean_fractional_entropy(y_amp), 0.8185810992233349,
                      atol=1e-10, rtol=1e-5)


def test_entropy_of_random_sparse():
    from mpsprep.helpers import entropy_of_random_sparse

    assert np.isclose(entropy_of_random_sparse(5, 0.5, 1, 1),
                      0.8185810992233349, atol=1e-10, rtol=1e-5)


def test_bit_string():
    from mpsprep.helpers import bit_string

    with pytest.raises(TypeError):
        bit_string(5, 1.)

    with pytest.raises(ValueError):
        bit_string(5, 32)


def test_update_kwargs_dict():
    from mpsprep.helpers import update_kwargs_dict

    assert update_kwargs_dict({"a": 1}, {"a": 2}) == {"a": 2}
    assert update_kwargs_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_state_fidelity():
    from mpsprep.helpers import state_fidelity

    state_a = np.array([
       0.27812066, 0.21679347, 0.08430580, 0.04019761, 0.10832718,
       0.10829279, 0.14375510, 0.24213786, 0.09394034, 0.23076171,
       0.21452612, 0.28698545, 0.08356753, 0.08362569, 0.01858036,
       0.26330896, 0.27219525, 0.08116547, 0.14862388, 0.25284245,
       0.20239129, 0.28364051, 0.08471362, 0.25304451, 0.12329953,
       0.09495510, 0.09167909, 0.00451785, 0.01753856, 0.22081281,
       0.19393188, 0.10685307,
       ]
       )
    state_b = np.array([
       0.00150442, 0.17158064, 0.36403395, 0.11368010, 0.16960558,
       0.01780508, 0.17952204, 0.35669648, 0.05243170, 0.21032557,
       0.00405388, 0.08185978, 0.11536375, 0.19619055, 0.10692718,
       0.12203119, 0.14609629, 0.14237095, 0.10683534, 0.10910380,
       0.08927364, 0.14472592, 0.11274467, 0.19183139, 0.29674203,
       0.29806470, 0.30851127, 0.11067829, 0.16802913, 0.03605062,
       0.12085971, 0.1814319,
       ]
       )

    assert np.isclose(state_fidelity(state_a, state_b),
                      0.49385457178029646, atol=1e-10, rtol=1e-5)
    assert np.isclose(state_fidelity(state_a, state_b),
                      state_fidelity(state_b, state_a), atol=1e-10, rtol=1e-5)
