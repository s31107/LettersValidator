import os

import Gui
import ImageLoader
import NetworkOfPerceptrons

train_data_dir = "./train_data"
eval_data = "./eval_data"
steepness_factor = 0.1
learning_rate = 0.1
init_layer_num = 1
number_of_perceptron_strategy = lambda x: 26
dimension = 100 * 100


perceptronNetwork = NetworkOfPerceptrons.PerceptronNetwork(init_layer_num, dimension, number_of_perceptron_strategy,
                                                           steepness_factor, learning_rate)
for label_name in os.listdir(train_data_dir):
    correct_output = [0] * 26
    correct_output[ord(label_name) - ord('A')] = 1
    for file in os.listdir(os.path.join(train_data_dir, label_name)):
        file_path = os.path.join(train_data_dir, label_name, file)
        perceptronNetwork.learn(ImageLoader.get_image_matrix(file_path, 100, 100), correct_output)

print(perceptronNetwork.compute(Gui.get_picture(100, 100)))

