import os
from datetime import datetime
from optimized_classifier_training import ModelTrainer

path = "C:\\Users\\User\\Desktop\\Coding\\CyPhIdv2"

for depth in range(7, 8):
    print(depth)
    time = datetime.now()
    time = time.strftime("%d-%m-%y_%Hh-%Mm")
    model_trainer = ModelTrainer()
    model_path = "models" + os.sep + "OptimizedModel"

    if os.path.exists(model_path):
        pass
    else:
        os.mkdir(model_path)


    #neurons_list = [[8, 8, 8, 8], [16, 16, 16, 16], [32, 32, 32, 32]]
    neurons_list = [[8, 8, 8, 16], [8, 16, 16, 16], [8, 8, 16, 32], [8, 16, 16, 32], [8, 16, 32, 32]]

    for neurons in neurons_list:
        model_trainer.run_trainer(path + os.sep + "wo_discarded_combined_X.p",
                                  path + os.sep + "wo_discarded_combined_y.p",
                                  depth,
                                  model_path + os.sep + "model_neurons_" + str(neurons) + "_" + time,
                                  neurons)

