from abc import ABC
import numpy as np
from scipy.stats.distributions import norm



"""
This module contains the definition of a base class for univariate distributions.
"""

class UnivariateDistribution(ABC):
    """
    This module contains the definition of a base class for univariate distributions.
    (Interface for discrete bounded uncertainty models assuming an equidistant grid)
    """
    def __init__(self, num_target_qubits, probabilities, low=0, high=1):
        """
        Abstract univariate distribution class
        Args:
            num_target_qubits (int): number of qubits it acts on
            probabilities (array or list):  probabilities for different states
            low (float): lower bound, i.e., the value corresponding to |0...0> (assuming an equidistant grid)
            high (float): upper bound, i.e., the value corresponding to |1...1> (assuming an equidistant grid)
        """
        #super().__init__(num_target_qubits)
        self.num_target_qubits = num_target_qubits
        self._num_values = 2 ** num_target_qubits
        self._probabilities = np.array(probabilities)
        self._low = low
        self._high = high
        self._values = np.linspace(low, high, self.num_values)
        if self.num_values != len(probabilities):
            raise ValueError('num qubits and length of probabilities vector do not match!')

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def num_values(self):
        return self._num_values

    @property
    def values(self):
        return self._values

    @property
    def probabilities(self):
        return self._probabilities

    def build(self, qc, q_ancillas=None):
        qc.initialize(self.probabilities/np.linalg.norm(self.probabilities), qc.qubits)
        return qc

    @staticmethod
    def pdf_to_probabilities(pdf, low, high, num_values):
        """
        Takes a probability density function (pdf), and returns a truncated and discretized array of probabilities corresponding to it
        Args:
            pdf (function): probability density function
            low (float): lower bound of equidistant grid
            high (float): upper bound of equidistant grid
            num_values (int): number of grid points
        Returns (list): array of probabilities
        """
        probabilities = np.zeros(num_values)
        values = np.linspace(low, high, num_values)
        total = 0
        for i, x in enumerate(values):
            probabilities[i] = pdf(x)
            total += probabilities[i]
        probabilities /= total
        return probabilities, values

    def pdf_to_probabilities_2(pdf, low, high, num_values):
        """
        Takes a probability density function (pdf), and returns a truncated and discretized array of squared probabilities corresponding to it
        Args:
            pdf (function): probability density function
            low (float): lower bound of equidistant grid
            high (float): upper bound of equidistant grid
            num_values (int): number of grid points
        Returns (list): array of probabilities
        """
        probabilities = np.zeros(num_values)
        values = np.linspace(low, high, num_values)
        total = 0
        for i,x in enumerate(values):
            probabilities[i] = pdf(x)**2
            total += probabilities[i]
        probabilities /= total
        return probabilities, values


"""
The Univariate Normal Distribution.
"""
class NormalDistributionWF(UnivariateDistribution):
    """
    The Univariate Normal Distribution.
    """

    CONFIGURATION = {
        'name': 'NormalDistribution',
        'description': 'Normal Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'NormalDistribution_schema',
            'type': 'object',
            'properties': {
                'num_target_qubits': {
                    'type': 'integer',
                    'default': 2,
                },
                'mu': {
                    'type': 'number',
                    'default': 0,
                },
                'sigma': {
                    'type': 'number',
                    'default': 1,
                },
                'low': {
                    'type': 'number',
                    'default': -1,
                },
                'high': {
                    'type': 'number',
                    'default': 1,
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_target_qubits, mu=0, sigma=1, low=-1, high=1):
        """
        Univariate normal distribution
        Args:
            num_target_qubits (int): number of qubits it acts on
            mu (float): expected value of considered normal distribution
            sigma (float): standard deviation of considered normal distribution
            low (float): lower bound, i.e., the value corresponding to |0...0> (assuming an equidistant grid)
            high (float): upper bound, i.e., the value corresponding to |1...1> (assuming an equidistant grid)
        """
        #self.validate(locals())
        probabilities, _ = UnivariateDistribution.\
            pdf_to_probabilities_2(lambda x: norm.pdf(x, mu, sigma), low, high, 2 ** num_target_qubits)
        super().__init__(num_target_qubits, probabilities, low, high)


