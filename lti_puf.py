import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from puf import Puf
from pulse import Pulse
from transfer_function import TransferFunction
import constants
from constants import c0

class LtiPuf(Puf):

    def __init__(self, h_f : TransferFunction, g_t : Pulse, challenge_length : int = 32, \
                 f_carrier : float = c0 / 1.55e-6, carrier_power : float = -10, carrier_RIN : float = -140, \
                 f_cutoff_pd : float = 50e9, pd_quantum_efficiency : float = 0.7, R_pd : float = 1e6,
                 input_refered_TIA_noise : float = 30e-12, \
                 noise_filter : npt.NDArray[np.floating] = None, \
                 sample_offset : int = None, samples_per_response_symbol : int = None, response_length : int = 32, \
                 response_pulse_duration : int = 16, off_value : complex = 1, \
                 return_analog : bool = False):
        """
        Optically encoded photonic PUF based on a linear token characterized by @h_f
        input challenge is modulated on pulses @g_t and comprises @challenge_length symbols

        laser diode characterized by: carrier frequency @f_carrier [Hz], power @carrier_power [dBm], relative intensity noise @carrier_RIN [dB / Hz]
        photo diode characterized by: 3dB cutoff frequency @f_cutoff_pd [Hz], quantum efficiency @pd_quantum_efficiency [%/100], shunt resistance @R_pd [Ohm] to compute the thermal noise due to the photodiode
        TIA stage following photodiode characterized by: @input_refered_TIA_noise [A / sqrt(Hz)]
        for our configuration typically noise due to TIA is dominating, shot and RIN noise are smaller and the thermal noise is almost negligible

        @noise_filter : filter that convovles output of photodiode current, it allows to improve the SNR by filtering noise outside of the signal bandwidth
        if None, the noise filter is 1/T_sample rect_{T_sample}(.) hence a filter averaging over one sample
        its physical nature could either be integrating behaviour of the "sample & hold" circuitry of the ADC or oversampling combined with a digital filter
        
        signal at output of photodiode is sampled with period @samples_per_response_symbol and offset @sample_offset
        in total @response_length symbols are sampled 

        the effect of @h_f on the challenge signal is computed by multiplication in the frequency domain
        however the DFT implies a periodic signal, while in reality the challenge signal is @off_value before and after the challenge symbols
        for a period of the discretized signal to correspond to the continuous signal we @off_value-pad our challenge signal
        we append @off_value samples for the duration of @response_pulse_duation symbols (the time for the pulse at the output to ring out)
        to obtain a good idea how to choose this @response_pulse_duation symbols you can plot @getOutputPulse()

        @return_analog defines wether to additionally return z(t), the analog signal from which the response is sampled 
        """
        super().__init__()
        self.g_t = g_t
        self.chal_len = challenge_length
        self.resp_len = response_length
        self.off_value_duration_sym = response_pulse_duration + (0 if noise_filter is None else int(np.ceil(len((noise_filter) - 1) / g_t.sps))) # we extend the region with @off_value to leverage circular convolution when applying @noise_filter
        self.samples = (challenge_length + self.off_value_duration_sym - 1) * self.g_t.sps + 1 + len(self.g_t.g_t) - 1
        self.h_f = h_f.resampleWindow(self.samples, f_carrier, 1 / g_t.t_sample)

        self.responsitivity = (pd_quantum_efficiency * constants.q) / (constants.h * f_carrier)
        self.psd_thermal_TIA = 4 * constants.k * constants.T0 / R_pd + input_refered_TIA_noise**2
        self.carrier_ampl = np.sqrt(10**(carrier_power / 10) * 1e-3) * np.sqrt(2) # not effective amplitude -> see LEN lecture notes
        self.carrier_RIN = 10**(carrier_RIN / 10)
        self.noise_filter = noise_filter

        # we model the photodiode frequency response as first order filter as we assume that the RC in the equivalent circuit dominates the frequency response
        # for more accurate modelling we would need to consider as well the drift and diffusion of the holes/electrons in the diode -> see OTR lecture notes
        # TODO maybe discuss with Paul about this?
        self.pd_filter = 1 / (1 + 1j * np.fft.rfftfreq(self.samples, g_t.t_sample) / f_cutoff_pd) # for z_t we have 2 * g_t.sps, see self.evaluate()
        
        if sample_offset == None:
            self.sample_offset = (len(self.g_t.g_t) +  (0 if noise_filter is None else len(noise_filter)) - 1) // 2 # center of symbol resulting from pulse and noise filter, TODO think if it would be better to implement this as center of symbol + group delay of (noise filter + h_f), in this go we could as well compute @response_pulse_duration from the impulse response corresponding to h_f
        else:
            self.sample_offset = sample_offset

        if samples_per_response_symbol == None:
            self.sprs = g_t.sps
        else:
            self.sprs = samples_per_response_symbol

        self.off_value = off_value

        self.return_analog = return_analog

    def evaluate(self, challenge : npt.NDArray[np.complexfloating]):
        # x(t) = Re{x_bb(t) * exp(j 2pi self.f_carrier t)}
        # the base band signal x_bb(t)
        x_t_bb = np.zeros((self.chal_len - 1) * self.g_t.sps + 1)
        x_t_bb[::self.g_t.sps] = challenge
        x_t_bb = np.convolve(x_t_bb, self.g_t.g_t, mode='full')
        x_t_bb = np.concatenate((x_t_bb, np.ones(self.g_t.sps * self.off_value_duration_sym) * self.off_value)) # TODO this might is fine for rectangular pulses, but for e.g. pulses with smooth transitions results in weird transitions -> think about a better solution 
        x_t_bb *= self.carrier_ampl

        # y(t) = h(t) convolve x(t) = Re{y_bb(t) * exp(j 2pi self.f_carrier t)}
        x_f_bb = np.fft.fftshift(np.fft.fft(x_t_bb)) 
        y_f_bb = self.h_f.h_f * x_f_bb
        y_t_bb = np.fft.ifft(y_f_bb) # we don't need the ifftshift as it only corresponds to modulation with exp(j ...) that disappears in the next step due to abs(...)
        
        # z(t) = LP{y(t)^2} = 1/2 |y_bb(t)|
        z_t_pd = self.responsitivity * np.abs(y_t_bb)**2 / 2 # photodiode reacts to power
        # apply low pass filter (electronic behaviour of photodiode)
        z_f = np.fft.rfft(z_t_pd) * self.pd_filter
        z_t = np.fft.irfft(z_f)

        # add noise 
        psd = self.psd_thermal_TIA + 2 * constants.q * z_t + z_t**2 * self.carrier_RIN # N_0, thermal & TIA + shot + RIN noise
        variance = psd / self.g_t.t_sample # noise variance per sample assuming that a sample corresponds to averaging the signal over the sample period, corresponds to applying a rectangular filter of length T_sample with height 1/T_sample, noise bandwidth follow from Satz von Parseval: df = int_0^T_sample |1/T_sample|^2 dt = 1/T_sample
        z_t += np.sqrt(variance) * np.random.randn(self.samples)
        # apply noise filter
        if not self.noise_filter is None:
            z_t_filtered = np.fft.irfft(np.fft.rfft(z_t) * np.fft.rfft(self.noise_filter, n=self.samples)) # we apply a circular convolution to see @off_value as well at the start of our signal
        else:
            z_t_filtered = z_t

        r = z_t_filtered[self.sample_offset:self.sample_offset + self.sprs * self.resp_len:self.sprs]

        if self.return_analog:
            return r, z_t_filtered, x_t_bb
        return r
    
    def getOutputPulse(self):
        g_f_hbp = np.fft.fftshift(np.fft.fft(self.g_t.g_t, n=self.samples))
        p_f = self.h_f.h_f * g_f_hbp
        p_t = np.fft.ifft(np.fft.ifftshift(p_f))
        return Pulse(p_t, self.g_t.sps, self.g_t.modulation_rate)