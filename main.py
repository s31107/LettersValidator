import pickle
from pathlib import Path

import Gui
from NetworkOfPerceptrons import PerceptronNetwork
import Trainer

perceptron_network_file: str = "./Perceptron.pn"
matrix_width: int = 32
matrix_height: int = 32


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
        print(Trainer.convert_result_from_perceptron(perceptron.compute(Gui.get_picture(matrix_width, matrix_height))))
else:
    perceptron: PerceptronNetwork = PerceptronNetwork(
        dimension=matrix_width * matrix_height,
        number_of_perceptron_strategy=[
            Trainer.output_dimension * 3, Trainer.output_dimension * 2, Trainer.output_dimension],
        learning_rate=0.6,
        steepness_factor=0.2
    )
    learned_perceptron: PerceptronNetwork = Trainer.train(perceptron, (matrix_width, matrix_height, ))
    serialize_strategy(learned_perceptron)
