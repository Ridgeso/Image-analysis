
from typing import Tuple
import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.models import Sequential, load_model, save_model


class Data:
    DataShape = (50, 50, 1)

    class Dataset:
        def __init__(self, data: np.array, labels: np.array) -> None:
            self.data = data
            self.labels = labels

    def __init__(self) -> None:
        self.trainingset = self.Dataset(np.zeros(0), np.zeros(0))
        self.testset = self.Dataset(np.zeros(0), np.zeros(0))
        self.isLoaded = False
    
    def loadData(self) -> None:
        data = []
        labels = []

        trainDataPath = Path.cwd() / "dataProcesing" / "Train"
        for dataset in os.listdir(trainDataPath):
            datasetPath = trainDataPath / dataset
            for imageName in os.listdir(datasetPath):
                try:
                    image = np.array(Image.open(datasetPath / imageName).resize(self.DataShape[:2]))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.adaptiveThreshold(
                        image,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        11,
                        2
                    )


                    data.append(image)
                    labels.append(dataset)
                    
                except UnidentifiedImageError as err:
                    print(f"Image {datasetPath / imageName} could not be loaded.\n{err}")
        
        data = np.array(data)
        labels = np.array(labels)
        
        trainsetData, testsetData, trainsetLabes, testsetLabes = train_test_split(
            data,
            labels,
            test_size=0.2,
            random_state=42
        )

        categoryNum = len(os.listdir(trainDataPath))
        trainsetLabes = to_categorical(trainsetLabes, categoryNum)
        testsetLabes = to_categorical(testsetLabes, categoryNum)

        self.trainingset = self.Dataset(trainsetData, trainsetLabes)
        self.trainingset = self.Dataset(testsetData, testsetLabes)
        self.isLoaded = True


class Model:
    ModelPath = Path.cwd() / "dataProcesing" / "roadSignsModel.h5"

    def __init__(self) -> None:
        self.model = None
        self.isTrained = False
    
    def loadModel(self, ofShape: Tuple[int, ...]) -> None:
        if self.ModelPath.exists():
            self.model = load_model(self.ModelPath)
            self.isTrained = self.model.layers[0].input_shape[1:] == ofShape
            if self.isTrained:
                return

        self.model = self._createModel(ofShape)
    
    def trainModel(self, epochs: int, data: Data) -> None:
        if not data.isLoaded:
            return

        self.model.fit(
            data.trainingset.data,
            data.trainingset.labels,
            batch_size=32,
            epochs=epochs,
            validation_data=(
                data.testset.data,
                data.testset.labels
            )
        )
        self.isTrained = True
    
    def saveModel(self) -> None:
        save_model(self.model, self.ModelPath)

    def predict(self, image: Image) -> int:
        image = image.resize(self.model.layers[0].input_shape[1:3])
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        image = np.reshape(image, (1, *self.model.layers[0].input_shape[1:3]))
        prediction = self.model.predict(image)
        return np.argmax(prediction)
    
    @staticmethod
    def _createModel(inputShape: Tuple) -> Sequential:
        model = Sequential()        
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=inputShape))
        model.add(Conv2D(filters=48, kernel_size=(4, 4), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.15))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.40))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(43, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


if __name__ == "__main__":
    epochs = 30
    data = Data()

    model = Model()
    print("Loading model")
    model.loadModel(data.DataShape)

    if not model.isTrained:
        print("Loading data")
        data.loadData()
        print("Training")
        model.trainModel(epochs, data)
        print("Saving")
        model.saveModel()

    print("Predicting")
    im = Image.open(Path.cwd() / "dataProcesing" / "Train" / "0" / "00000_00002_00010.png")
    print(model.predict(im))
