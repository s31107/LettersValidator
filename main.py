import pickle
from pathlib import Path

import Gui
from NetworkOfPerceptrons import PerceptronNetwork, normalize_vector
import Trainer

perceptron_network_file: str = "./Perceptron.pn"
matrix_width: int = 64
matrix_height: int = 64


def serialize_strategy(obj: PerceptronNetwork) -> None:
    with open(perceptron_network_file, "wb") as file:
        pickle.dump(obj, file)


def deserialize_strategy() -> PerceptronNetwork:
    with open(perceptron_network_file, "rb") as file:
        return pickle.load(file)


if Path(perceptron_network_file).is_file():
    perceptron: PerceptronNetwork = deserialize_strategy()
    Trainer.evaluate_perceptron_network(
        perceptron, Trainer.get_all_images_with_result(Trainer.eval_data, (matrix_width, matrix_height, )))
    while True:
        print(Trainer.convert_result_from_perceptron(
            perceptron.compute(normalize_vector(Gui.get_picture(matrix_width, matrix_height)))))
else:
    perceptron: PerceptronNetwork = PerceptronNetwork(
        init_layer_number=9,
        dimension=matrix_width * matrix_height,
        number_of_perceptron_strategy=lambda layer, layers_number: Trainer.output_dimension
            if layer == layers_number - 1 else (layers_number - layer) * Trainer.output_dimension,
        learning_rate=0.3,
        steepness_factor=0.2
    )
    learned_perceptron: PerceptronNetwork = Trainer.train(perceptron, (matrix_width, matrix_height, ))
    serialize_strategy(learned_perceptron)