import numpy as np

from simulator.grid.indexing import wrap_i, neigh_i, neigh_j, pad_halo


def test_wrap_and_neighbors():
    nx, ny = 10, 5
    assert wrap_i(-1, nx) == nx - 1
    assert wrap_i(nx, nx) == 0

    im1, i, ip1 = neigh_i(0, nx)
    assert (im1, i, ip1) == (nx - 1, 0, 1)

    jm1, j, jp1 = neigh_j(0, ny)
    assert (jm1, j, jp1) == (0, 0, 1)

    jm1, j, jp1 = neigh_j(ny - 1, ny)
    assert (jm1, j, jp1) == (ny - 2, ny - 1, ny - 1)


def test_pad_halo_shapes_and_edges():
    A = np.arange(12, dtype=float).reshape(3, 4)
    B = pad_halo(A, halo=1, periodic_i=True)
    # Shape grows by 2 in both dims
    assert B.shape == (5, 6)
    # Corners replicate edges correctly (periodic in i, clamp in j)
    assert np.isclose(B[0, 0], B[1, -2])  # top-left equals element above left-edge wrapping