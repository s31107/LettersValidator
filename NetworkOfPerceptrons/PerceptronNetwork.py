from typing import Callable

from NetworkOfPerceptrons.PerceptronLayer import PerceptronLayer


class PerceptronNetwork:
    def __init__(self, init_layer_number: int, dimension: int, number_of_perceptron_strategy: Callable[[int, int], int],
                 steepness_factor: float, learning_rate: float):
        self.perceptron_layers = []
        last_perceptron_number = dimension
        for index in range(init_layer_number):
            current_perceptron_number: int = number_of_perceptron_strategy(index, init_layer_number)
            self.perceptron_layers.append(PerceptronLayer(current_perceptron_number, last_perceptron_number,
                                                          steepness_factor, learning_rate))
            last_perceptron_number: int = current_perceptron_number

    def learn(self, input_vector: list[float], correct_output: list[float]) -> None:
        perceptron_output: list[list[float]] = [input_vector]
        for index in range(len(self.perceptron_layers)):
            perceptron_output.append(self.perceptron_layers[index].compute(perceptron_output[-1]))

        last_error_signals: list[float] = self.perceptron_layers[-1].get_error_signals_for_last_layer(
            perceptron_output[-1], correct_output)
        self.perceptron_layers[-1].learn(perceptron_output[-2], last_error_signals)
        for index in range(len(self.perceptron_layers) - 2, -1, -1):
            last_error_signals: list[float] = self.perceptron_layers[index].get_error_signals_for_hide_layer(
                perceptron_output[index + 1], last_error_signals, self.perceptron_layers[index + 1].get_weights())
            self.perceptron_layers[index].learn(perceptron_output[index], last_error_signals)

    def compute(self, input_vector: list[float]) -> list[float]:
        last_output: list[float] = input_vector
        for perceptron_layer in self.perceptron_layers:
            last_output: list[float] = perceptron_layer.compute(last_output)
        return last_output

    def normalize_network_weights(self):
        for item in self.perceptron_layers:
            item.normalize_layer_weights()
