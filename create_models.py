import os
from datetime import datetime
from classifier_trainer import ModelTrainer

path = "C:\\Users\\User\\Desktop\\Coding\\CyPhIdv2"

for depth in range(1, 12):
    print(depth)
    time = datetime.now()
    time = time.strftime("%d-%m-%y_%Hh-%Mm")
    model_trainer = ModelTrainer()
    model_path = "models" + os.sep + "ValLossCheckpointPatience50"

    if os.path.exists(model_path):
        pass
    else:
        os.mkdir(model_path)

    model_trainer.run_trainer(path + os.sep + "wo_discarded_combined_X.p",
                              path + os.sep + "wo_discarded_combined_y.p",
                              depth,
                              model_path + os.sep + "model_depth_" + str(depth) + "_" + time)

