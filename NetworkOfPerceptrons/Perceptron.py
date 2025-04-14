import random
import math

_init_compartment = (-2, 2, )


def compute_error_signal_for_last_layer(computed_input: float, correct_output: float) -> float:
    assert 0 <= correct_output <= 1
    return computed_input * (1 - computed_input) * (correct_output - computed_input)


def compute_error_signal_for_hide_layer(computed_input: float,
                                        error_signals_from_other_layer: list[tuple[float, float]]) -> float:
    return computed_input * (1 - computed_input) * sum(item[0] * item[1] for item in error_signals_from_other_layer)


class Perceptron:
    def __init__(self, dimension: int, steepness_factor: float, learning_rate: float):
        assert 0 < steepness_factor <= 1
        assert 0 < learning_rate <= 1
        assert dimension >= 1
        self._weights = [random.uniform(*_init_compartment) for _ in range(dimension)]
        self._threshold = random.uniform(*_init_compartment)
        self._steepness_factor: float = steepness_factor
        self._learning_rate: float = learning_rate

    def get_weights(self) -> list[float]:
        return self._weights

    def compute(self, vector_input: list[float]) -> float:
        assert len(vector_input) == len(self._weights)
        net: float = sum(vector_input[i] * self._weights[i] for i in range(len(vector_input))) - self._threshold
        return 1 / (1 + math.exp(-self._steepness_factor * net))

    def learn(self, input_vector: list[float], error_signal: float) -> None:
        assert len(input_vector) == len(self._weights)
        for i in range(len(self._weights)):
            self._weights[i] += self._learning_rate * input_vector[i] * error_signal
        self._threshold -= self._learning_rate * error_signal
