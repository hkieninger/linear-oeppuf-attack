import numpy as np
import numpy.typing as npt
import rrc

class Pulse:

    def __init__(self, g_t : npt.NDArray[np.floating | np.complexfloating], samples_per_symbol : int, modulation_rate : float):
        self.g_t = g_t
        self.sps = samples_per_symbol
        self.modulation_rate = modulation_rate
        self.t_sample = 1 / modulation_rate / samples_per_symbol
        pass

    @classmethod
    def raisedCosinePulse(cls, samples : int = 8 * 64 + 1, samples_per_symbol : int = 64, rolloff : float = 0.5, modulation_rate : float = 10e9):
        '''
        constructs a raised cosine pulse
        '''
        if samples % 2 != 1:
            raise ValueError("@samples must be odd, but is even!")
        g_t = rrc.get_rc_ir(samples, samples_per_symbol, 1, rolloff)
        g_t /= np.max(g_t) # normalize pulse amplitude
        return cls(g_t, samples_per_symbol, modulation_rate)
    
    @classmethod
    def rectPulse(cls, samples_per_symbol : int = 64, modulation_rate : float = 10e9):
        return cls(np.ones(samples_per_symbol), samples_per_symbol, modulation_rate)