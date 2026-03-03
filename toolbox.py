import numpy as np
import numpy.typing as npt

def transfer_to_scattering_matrix(T : npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """
    @return scattering matrix(es) (shape (...,2,2)) corresponding to transfer matrix(es) @T (shape (...,2,2))
    """
    t11 = T[..., 0, 0]
    t12 = T[..., 0, 1]
    t21 = T[..., 1, 0]
    t22 = T[..., 1, 1]

    S = np.empty(T.shape, dtype=complex)
    S[..., 0, 0] = t12 / t22
    S[..., 0, 1] = (t11 * t22 - t12 * t21) / t22
    S[..., 1, 0] = 1.0 / t22
    S[..., 1, 1] = -t21 / t22

    return S