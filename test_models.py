import os
from classifier import Classifier

path = "C:\\Users\\User\\Desktop\\Coding\\CyPhIdv2"

data = "Model;Microscope;Phase;Total;P1;P2;P3;\n"

classifier = Classifier()

for model in os.listdir("models\\OptimizedModel"):
    classifier.load_model("models\\OptimizedModel" + os.sep + model)

    for microscope in ["Arara", "Elyra"]:
        for phase in ["1", "2", "3"]:
            classifier.load_data(data_path=path + os.sep + microscope + " Pickles" + os.sep + "test_X_phase" + phase + ".p",
                                 microscope=microscope)
            preds = classifier.classify_data()
            total = len(preds)
            p1 = 0
            p2 = 0
            p3 = 0

            for pred in preds:
                if pred == 0:
                    p1 += 1
                elif pred == 1:
                    p2 += 1
                elif pred == 2:
                    p3 += 1
                else:
                    print("Bugged pred handling")

            data += model + ";" + microscope + ";Phase " + phase + ";" + str(total) + ";" + str(p1) + ";" + str(p2) + ";" + str(p3) + ";\n"

open("ablation_results_neurons_v2.csv", "w").writelines(data)

