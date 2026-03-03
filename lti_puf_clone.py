import numpy as np
import numpy.typing as npt

from puf import Puf

class LtiPufClone(Puf):

    def __init__(self, challenges : npt.NDArray[np.floating], responses : npt.NDArray[np.floating], response_pulse_duration : int, sample_offset : int, samples_per_challenge_symbol : int, samples_per_response_symbol : int, off_value : float = 1):
        """
        @challenges.shape (crp_count, challenge_length)
        @responses.shape (crp_count, response_length)
        @response_pulse_duration the duration of the pulse in symbols until it has decayed to zero
        """
        crp_count, challenge_length = challenges.shape
        _, self.response_length = responses.shape
        self.response_pulse_duration = response_pulse_duration
        self.sample_offset = sample_offset
        self.samples_per_response_symbol = samples_per_response_symbol
        self.samples_per_challenge_symbol = samples_per_challenge_symbol
        self.off_value = off_value

        self.parameters = np.empty((self.response_length, response_pulse_duration * response_pulse_duration), dtype=np.float64)
        challenge_symbols_offset = np.concatenate((np.ones((crp_count, response_pulse_duration - 1)) * off_value, challenges), axis=1) # we can model the input signal before the challenge is applied as pulses (e.g. of rectangular shape) with amplitude @off_value (e.g. 0)
        for l in range(self.response_length):
            t = sample_offset + l * samples_per_response_symbol
            k = int(np.floor(t / samples_per_challenge_symbol)) # causality, maximum k that affects response


            c = challenge_symbols_offset[:,k:k + response_pulse_duration] # shape (crp_count, response_pulse_duration), all challenge symbols that affect response symbol l
            r = responses[:,l] # shape (crp_count)

            c_monomials = (c[:,:,np.newaxis] * np.conj(c[:,np.newaxis,:])).reshape(crp_count, -1)
            self.parameters[l] = np.linalg.pinv(c_monomials) @ r

    def evaluate(self, challenge):
        response = np.empty(self.response_length, dtype=float)

        challenge_symbols_offset = np.concatenate((np.ones((self.response_pulse_duration - 1) * self.off_value), challenge))
        for l in range(self.response_length):
            t = self.sample_offset + l * self.samples_per_response_symbol
            k = int(np.floor(t / self.samples_per_challenge_symbol)) # causality, maximum k that affects response

            c = challenge_symbols_offset[k:k + self.response_pulse_duration] # shape (response_pulse_duration), all challenge symbols that affect response symbol l

            c_monomials = np.outer(c,np.conj(c)).flatten()
            response[l] = np.inner(self.parameters[l], c_monomials)

        return response