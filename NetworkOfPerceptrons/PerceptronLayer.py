from NetworkOfPerceptrons import Perceptron


class PerceptronLayer:
    def __init__(self, number_of_perceptron: int, dimension: int, steepness_factor: float, learning_rate: float):
        assert 0 < number_of_perceptron
        self.perceptron_layer = [Perceptron.Perceptron(dimension, steepness_factor, learning_rate)
                                 for _ in range(number_of_perceptron)]

    def get_weights(self) -> list[list[float]]:
        return [item.get_weights() for item in self.perceptron_layer]

    def compute(self, vector_input: list[float]) -> list[float]:
        return [perceptron.compute(vector_input) for perceptron in self.perceptron_layer]

    def get_error_signals_for_last_layer(self, computed_input: list[float],
                                         correct_outputs: list[float]) -> list[float]:
        assert len(correct_outputs) == len(self.perceptron_layer)
        return [Perceptron.compute_error_signal_for_last_layer(computed_input[index], correct_outputs[index])
                for index in range(len(self.perceptron_layer))]

    def get_error_signals_for_hide_layer(self, computed_input: list[float], error_signals_from_other_layer: list[float],
                                         weights_from_other_layer: list[list[float]]) -> list[float]:
        assert len(error_signals_from_other_layer) == len(weights_from_other_layer)
        return [
            Perceptron.compute_error_signal_for_hide_layer(
                computed_input[index], list(zip(error_signals_from_other_layer, (
                    item[index] for item in weights_from_other_layer))))
            for index in range(len(self.perceptron_layer))]

    def learn(self, input_vector: list[float], error_signals: list[float]) -> None:
        for index in range(len(self.perceptron_layer)):
            self.perceptron_layer[index].learn(input_vector, error_signals[index])

    def normalize_layer_weights(self):
        for item in self.perceptron_layer:
            item.normalize_weights()
