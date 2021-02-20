from imageai.Detection.Custom import DetectionModelTrainer

Model_trainer = DetectionModelTrainer()
Model_trainer.setModelTypeAsYOLOv3()
Model_trainer.setDataDirectory(data_directory="apple_dataset")
Model_trainer.setTrainConfig(object_names_array=["apple", "damaged_apple"], batch_size=8, num_experiments=50, train_from_pretrained_model="pretrained-yolov3.h5")
Model_trainer.trainModel()