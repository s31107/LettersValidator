import random

import ImageLoader
from NetworkOfPerceptrons import PerceptronNetwork
import os

acc: float = 91.
train_data_dir = "./train_data"
eval_data = "./eval_data"
output_dimension = ord("Z") + 1 - ord("A")


def get_all_images_with_result(path: str, size: tuple[int, int]) -> list[tuple[str, list[float]]]:
    list_files = []
    for label_name in os.listdir(path):
        for file in os.listdir(path + "/" + label_name):
            list_files.append(
                (label_name, ImageLoader.get_image_matrix(os.path.join(path, label_name, file), size)))
    return list_files


def train(perceptron_network: PerceptronNetwork, size: tuple[int, int]) -> PerceptronNetwork:
    all_train_files: list[tuple[str, list[float]]] = get_all_images_with_result(train_data_dir, size)
    all_eval_files: list[tuple[str, list[float]]] = get_all_images_with_result(eval_data, size)
    computed_acc: int = 0
    while computed_acc <= acc:
        # Learning:
        random.shuffle(all_train_files)
        for train_unit in all_train_files:
            correct_output: list[int] = [0] * output_dimension
            correct_output[ord(train_unit[0]) - ord('A')] = 1
            perceptron_network.learn(train_unit[1], correct_output)
        # Evaluating:
        computed_acc = evaluate_perceptron_network(perceptron_network, all_eval_files)
    return perceptron_network


def evaluate_perceptron_network(
        perceptron_network: PerceptronNetwork, all_eval_files: list[tuple[str, list[float]]]) -> int:
    computed_acc: int = 0
    random.shuffle(all_eval_files)
    for eval_unit in all_eval_files:
        computed_letter: str = convert_result_from_perceptron(perceptron_network.compute(eval_unit[1]))
        computed_acc += int(computed_letter == eval_unit[0])
    computed_acc = (computed_acc / len(all_eval_files)) * 100
    print(f"Epoch acc = {computed_acc}")
    return computed_acc


def convert_result_from_perceptron(data: list[float]) -> str:
    return chr(ord("A") + data.index(max(data)))