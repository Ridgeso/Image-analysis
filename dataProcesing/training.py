
from typing import Tuple
import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from sklearn.model_selection import train_test_split

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.models import Sequential, load_model, save_model


class Data:
    DataShape = (30, 30, 3)
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

        trainDataPath = Path.cwd() / "Train"
        for dataset in os.listdir(trainDataPath):
            datasetPath = trainDataPath / dataset
            for imageName in os.listdir(datasetPath):
                try:
                    image = np.array(Image.open(datasetPath / imageName).resize(self.DataShape[:2]))

                    data.append(image)
                    labels.append(dataset)
                    
                except UnidentifiedImageError as err:
                    print(f"Image {datasetPath / imageName} could not be loaded.\n{err}")
        
        data = np.array(self.data)
        labels = np.array(self.labels)
        
        trainsetData, testsetData, trainsetLabes, testsetLabes = train_test_split(
            self.data,
            self.labels,
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
    ModelPath = Path.cwd() / "roadSignsModel.h5"

    def __init__(self, epochs: int, trainingData) -> None:
        self.model = None
        self.isTrained = False
        self.epochs = epochs
        self.trainingData = trainingData
    
    def loadModel(self) -> Sequential:
        if self.ModelPath.exists():
            model = load_model(self.ModelPath)
            if model.layers[0].input_shape[1:] == self.trainingData.DataShape:
                self.isTrained = True
                return model
        
        model = self._createModel(self.trainingData.DataShape)
        return model
    
    def trainModel(self) -> None:
        if not self.trainingData.isLoaded:
            self.trainingData.loadData()

        self.model.fit(
            self.trainingData.trainingset.data,
            self.trainingData.trainingset.labels,
            batch_size=32,
            epochs=self.epochs,
            validation_data=(
                self.trainingData.testset.data,
                self.trainingData.testset.labels
            )
        )
        self.isTrained = True
    
    def saveModel(self) -> None:
        save_model(self.model, self.ModelPath)
    
    @staticmethod
    def _createModel(inputShape: Tuple) -> Sequential:
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=inputShape))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(43, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

if __name__ == "__main__":
    data = Data()

    model = Model(15, data)
    model.loadModel()

    if not model.isTrained:
        model.trainModel()
        model.saveModel()
