from classifier_training import ModelTrainer

model = ModelTrainer()
model.run_trainer("X.p",
                  "y.p")
model.save_model(path="model")

