import random

import ImageLoader
from NetworkOfPerceptrons import PerceptronNetwork
import os

acc: float = 91.
train_data_dir = "./train_data"
eval_data = "./eval_data"

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
        computed_acc: int = 0
        # Learning:
        random.shuffle(all_train_files)
        for train_unit in all_train_files:
            correct_output: list[int] = [0] * 26
            correct_output[ord(train_unit[0]) - ord('A')] = 1
            perceptron_network.learn(train_unit[1], correct_output)
        # Evaluating:
        random.shuffle(all_eval_files)
        for eval_unit in all_eval_files:
            computed_letter: str = convert_result_from_perceptron(perceptron_network.compute(eval_unit[1]))
            computed_acc += int(computed_letter == eval_unit[0])
        computed_acc /= 100 / len(all_eval_files)
        print(f"Epoch acc = {computed_acc}")
    return perceptron_network

def convert_result_from_perceptron(data: list[float]) -> str:
    return chr(ord("A") + data.index(max(data)))