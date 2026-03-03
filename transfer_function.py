import math
import numpy as np
import numpy.typing as npt
import scipy
import logging
logger = logging.getLogger(__name__)

from constants import c0
import toolbox

class TransferFunction:

    def __init__(self, start_freq : float, stop_freq : float, h_f : npt.NDArray[np.complexfloating]):
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.h_f = h_f
        self.freqs = np.linspace(start_freq, stop_freq, len(h_f))

    def save_npz(self, filename: str):
        """Save object data to a compressed NumPy .npz file."""
        np.savez(filename,
                 start_freq=self.start_freq,
                 stop_freq=self.stop_freq,
                 h_f=self.h_f)

    @classmethod
    def load_npz(cls, filename: str):
        """Load a TransferFunction (or subclass) instance from a .npz file."""
        data = np.load(filename)
        return cls(
            float(data['start_freq']),
            float(data['stop_freq']),
            data['h_f']
        )
    
    def resampleWindow(self, samples : int, f_center : float, bandwidth : float):
        """
        constructs a resampled TransferFunction with @samples from the window around @f_center with width @bandwidth
        """
        if f_center - bandwidth / 2 < self.start_freq or f_center + bandwidth / 2 > self.stop_freq :
            raise ValueError("Window given by @f_center and @bandwidth larger than frequency range of current TransferFunction!")

        current_res = (self.stop_freq - self.start_freq) / (len(self.h_f) - 1)
        new_res = bandwidth / (samples - 1)
        if(new_res < current_res):
            logger.debug(f"resampling: new resolution ({new_res}) lower than current resolution ({current_res}), make sure that no spectral features are lost!")

        window = np.logical_and(f_center - bandwidth / 2 <= self.freqs, self.freqs <= f_center + bandwidth / 2)
        resampled = scipy.signal.resample(self.h_f[window], samples)
        return TransferFunction(f_center - bandwidth / 2, f_center + bandwidth / 2, resampled)

    @classmethod
    def random(cls, start_wavelength : float = 1.5e-6, stop_wavelength : float = 1.6e-6, wavelength_resolution : float = 1e-12, filter_sigma : float = 300):
        """
        construct random transfer function in the range between @start_wavelength and @stop_wavelength with approx. @wavelength_resolution
        the samples are randomly drawn from a normal distribution and then filtered with a guassian filter characterized by @filter_sigma
        """
        start_freq = c0 / stop_wavelength # [Hz]
        stop_freq = c0 / start_wavelength # [Hz]
        center_freq = (start_freq + stop_freq) / 2 # [Hz]
        freq_resolution = center_freq**2 / c0 * wavelength_resolution # [Hz]

        point_count = round((stop_freq - start_freq) / freq_resolution)
        logger.debug(f"Random transfer function with {point_count} points will be constructed.")
        h_f = scipy.ndimage.gaussian_filter1d(np.random.randn(point_count) +  1j * np.random.randn(point_count), sigma=filter_sigma)
        return cls(start_freq, stop_freq, h_f)
    
    @classmethod
    def moireGrating(cls, start_wavelength : float = 1.53e-6, stop_wavelength : float = 1.531e-6, wavelength_resolution : float = 1e-12, \
                     Lambda1 : float = 316e-9, Lambda2 : float = 317e-9, length : float = 500e-6, \
                     f0 = c0 / 1.55e-9, neff0 = 2.444, \
                     f1 = c0 / 1.55e-9, neff1 = 2.411, \
                     f2 = c0 / 1.55e-9, neff2 = 2.474, \
                     delta_neff : float = 8e-3, delta_L : float = 1e-9, loss_dB_per_cm : float = 2.4):
        """
        constructs a transfer function from a moire grating considering random manufacturing variations (similar as described in Tarik 2020)
        the transfer function is obtained with the Transfer Matrix Method (similar to Lumerical EME, but not considering the field distribution)

        @neff0: effective index of input and output waveguides
        @neff1/@neff2: alternating effective indices in the grating
        if @neffi is an array @fi describes the frequencies where the effective indices have been sampled

        considered variation are
        - width and height of waveguide segements resulting in variations of @neff1/@neff, described by uniform distribution [-delta_neff, +delta_neff]
        - roughness of waveguide segments walls resulting in variations of loss, described by uniform distribution [0, alpha_loss * 2] => E(...) = alpha_loss
        - length of waveguide segments, described by uniform distribution [-delta_L, +delta_L]
        while variations in n_eff affect amplitude of resonances, variations in segment length affect the frequency of resonances

        the longer @length the smaller the feature size in @return (e.g; length=500um -> feature size approx. 1nm, length=1cm -> feature size approx. 20pm)
        """

        # transform wavelength parameters into freq parameters
        start_freq = c0 / stop_wavelength # [Hz]
        stop_freq = c0 / start_wavelength # [Hz]
        center_freq = (start_freq + stop_freq) / 2 # [Hz]
        freq_resolution = center_freq**2 / c0 * wavelength_resolution # [Hz]

        point_count = round((stop_freq - start_freq) / freq_resolution)
        logger.debug(f"Random transfer function with {point_count} points will be constructed.")

        # interpolate effective indices onto desired frequencies
        f = np.linspace(start_freq, stop_freq, point_count)
        neff0 = linear_interpolation(f, f0, neff0)
        neff1 = linear_interpolation(f, f1, neff1)
        neff2 = linear_interpolation(f, f2, neff2)

        # compute and randomize moire grating segments
        Ls = moireGratingSegments(Lambda1, Lambda2, length)
        Ls += np.random.uniform(-1, 1, size=len(Ls)) * delta_L # TODO think about type of distribution to use (condition on sum of variations should be zero?)
        Ls[Ls < 0 ] = 0 # segments with negative length are assumed to result in no segment

        neffs = np.empty((len(Ls), point_count)) # build randomly by interpolating neff for waveguide width
        neffs[0::2] = neff1
        neffs[1::2] = neff2
        neffs[:] += np.random.uniform(-1, 1, size=len(Ls)).reshape(len(Ls),1) * delta_neff # TODO think about type of distribution to use (proposition from Tarik 2020)

        alpha = np.log(10) * loss_dB_per_cm * 100 / 20
        delta_alpha = alpha / 2 
        alphas = np.random.uniform(size=len(Ls)) * delta_alpha # TODO think about type of distribution to use (proposition from Tarik 2020)

        # compute the transfer matrix of the moire grating
        T_moire = transfer_matrix_interface(neff0, neffs[0]).astype(np.complex128)
        T_moire @= transfer_matrix_propagation(neffs[0], f, Ls[0], alphas[0])
        for s in range(1, len(Ls)):
            T_moire @= transfer_matrix_interface(neffs[s-1], neffs[s])
            T_moire @= transfer_matrix_propagation(neffs[s], f, Ls[s], alphas[s])
        T_moire @= transfer_matrix_interface(neffs[-1], neff0)

        # compute the scattering parameter of interest: s_11
        S = toolbox.transfer_to_scattering_matrix(T_moire)
        return cls(start_freq, stop_freq, S[:,0,0]) # s_21 has a large disturbing DC part that carries no information of the variations

# =============================================================================
# Helper functions
# =============================================================================
def linear_interpolation(f, f_src, neff_src):
    """
    linearly interpolates @neff_src sampled at @f_src onto points @f
    @f_src can be either an ascending or descending array
    if neff_src is a scalar it is returned as scalar
    """
    if np.isscalar(f_src):
        return neff_src
    if f_src[-1] < f_src[0]:
        f_src, neff_src = f_src[::-1], neff_src[::-1]
    return np.interp(f, f_src, neff_src)

def transfer_matrix_interface(neff_left, neff_right):
    """
    @return the transfer matrix at a 1D interface between @neff_left and @neff_right
    """
    # aus Saleh und Teich / OWF Uebungsblatt 3 / Tarik 2020 supplementary material entnommen
    t22 = (neff_left + neff_right) / (2 * neff_right)
    t12 = (neff_right - neff_left) / (2 * neff_right)
    return np.array([
        [t22, t12],
        [t12, t22]
    ]).transpose(2,0,1) # shape (*,2,2)

def transfer_matrix_propagation(neff, f, L, alpha):
    """
    @return the transfer matrix for propagation through medium @neff of length @L with attenuation @alpha for frequencies @f
    """
    # aus Saleh und Teich / OWF Uebungsblatt 3 / Tarik 2020 supplementary material entnommen
    t22 = np.exp(1j * 2 * np.pi * f / c0 * neff * L + alpha * L)
    return np.array([
        [t22**-1, np.zeros(len(t22))], 
        [np.zeros(len(t22)), t22]
    ]).transpose(2,0,1) # shape (*,2,2)

def moireGratingSegments(Lambda1 = 316e-9, Lambda2 = 317e-9, L = 500e-6, resolution=1e-9):
    """
    @return an array of the segment lengths of a moire grating as described in Tarik 2020
    @Lambda1, @Lambda2 : periodicity of ANDed gratings
    @L : total length of the grating
    @resolution : minimum length of a segment

    REMARK: implementation has not been tested for edge cases
    """
    Lambda1_int = int(Lambda1 / resolution) # nm
    Lambda2_int = int(Lambda2 / resolution) # nm
    gcd =  math.gcd(Lambda1_int, Lambda2_int)
    dz = gcd / 2 # nm
    period = Lambda1_int * Lambda2_int / gcd

    z = np.arange(0, period, dz)
    grating1 = (z % Lambda1_int) < (Lambda1_int / 2)
    grating2 = (z % Lambda2_int) < (Lambda2_int / 2)
    moire = np.logical_and(grating1, grating2)
    moire_idx, = np.nonzero(np.diff(moire))
    segments = np.diff(moire_idx+1, prepend=0, append=len(moire)) * dz

    L_res = L / resolution
    periods_count = int(L_res // period)
    period_remainder = L_res % period
    remainder_segments = segments[np.cumsum(segments) < period_remainder]
    segments = np.concatenate((np.tile(segments, periods_count), remainder_segments))
    segments *= resolution
    return np.append(segments, L - np.sum(segments))