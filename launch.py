from typing import Callable
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np

from dataProcesing import Data, Model


class RoadSignGuesser:
    SCREEN_WIDTH = 1080
    SCREEN_HEIGHT = 720
    BACKGROUND_COLOR = "#090960"
    FOREGROUND_COLOR = "#63768D"
    BUTTONS_COLOR = "#8AC6D0"
    FONT_COLOR = "#FFFFFF"
    RoadSignsTransations = {
        0: 'Ograniczenie prędkości (20 km/h)',
        1: 'Ograniczenie prędkości (30 km/h)',
        2: 'Ograniczenie prędkości (50 km/h)',
        3: 'Ograniczenie prędkości (60 km/h)',
        4: 'Ograniczenie prędkości (70 km/h)',
        5: 'Ograniczenie prędkości (80 km/h)',
        6: 'Koniec ograniczenia prędkości (80 km/h)',
        7: 'Ograniczenie prędkości (100 km/h)',
        8: 'Ograniczenie prędkości (120 km/h)',
        9: 'Zakaz wyprzedzania',
        10: 'Zakaz wyprzedzania pojazdów o masie powyżej 3,5 tony',
        11: 'Pierwszeństwo na skrzyżowaniu',
        12: 'Droga z pierwszeństwem',
        13: 'Ustąp pierwszeństwa',
        14: 'Stop',
        15: 'Zakaz ruchu',
        16: 'Zakaz ruchu pojazdów o masie powyżej 3,5 tony',
        17: 'Zakaz wjazdu',
        18: 'Niebezpieczeństwo',
        19: 'Ostry zakręt w lewo',
        20: 'Ostry zakręt w prawo',
        21: 'Ostre zakręty',
        22: 'Wyboje',
        23: 'Śliska jezdnia',
        24: 'Zwężenie z prawej strony',
        25: 'Roboty drogowe',
        26: 'Sygnalizacja świetlna',
        27: 'Piesi',
        28: 'Przejście dla dzieci',
        29: 'Przejście dla rowerzystów',
        30: 'Uwaga na gołoledź',
        31: 'Uwaga dzikie zwierzęta',
        32: 'Koniec zakazów',
        33: 'Nakaz skrętu w prawo',
        34: 'Nakaz skrętu w lewo',
        35: 'Nakaz jazdy prosto',
        36: 'Nakaz jazdy prosto lub w prawo',
        37: 'Nakaz jazdy prosto lub w lewo',
        38: 'Nakaz jazdy z prawej strony znaku',
        39: 'Nakaz jazdy z lewej strony znaku',
        40: 'Skrzyżowanie o ruchu okrężnym',
        41: 'Koniec zakazu wyprzedzania',
        42: 'Koniec zakazu wyprzedzania pojazdów o masie powyżej 3,5 tony'
    }

    def __init__(self) -> None:
        self.model = Model()
        self.model.loadModel(Data.DataShape)

        self.root = tk.Tk()
        self.root.title("Rozpoznaj Znak")
        self.root.geometry(f"{self.SCREEN_WIDTH}x{self.SCREEN_HEIGHT}")
        self.root.configure(bg=self.BACKGROUND_COLOR)

        self.appFrame = tk.Frame(self.root, bg=self.BACKGROUND_COLOR)
        self.appFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._AddSpacing()

        self.loadButton = tk.Button(
            self.root,
            text="Wybierz zdjęcie",
            command=self.LoadPhoto,
            bg=self.BUTTONS_COLOR,
            fg=self.FONT_COLOR,
            relief=tk.RAISED,
            bd=2,
            font=("Helvetica", 16)
        )
        self.loadButton.pack(in_=self.appFrame)

        self._AddSpacing()
        
        self.imageFrame = tk.Frame(self.appFrame, bg=self.BACKGROUND_COLOR)
        self.imageFrame.pack(side=tk.TOP)

        self.imageLabel = tk.Label(self.root, background=self.BACKGROUND_COLOR)
        self.imageLabel.pack(in_=self.imageFrame, side=tk.LEFT)

        self.signOryPhoto = self._GetImage(Path.cwd() / "assets" / "empty.png")
        self.signPhoto = ImageTk.PhotoImage(self.signOryPhoto)
        self.imageLabel.config(image=self.signPhoto)

        self._AddSpacing(tk.LEFT, self.imageFrame)

        self.sliderStyle = ttk.Style()
        self.sliderStyle.configure(
            "Vertical.TScale",
            sliderlength=50,
            troughcolor="#090960",
            sliderthickness=10,
            background=self.FOREGROUND_COLOR
        )

        self.kontrastFrame = tk.Frame(self.imageFrame, background=self.FOREGROUND_COLOR)
        self.kontrastFrame.pack(side=tk.LEFT)
        
        self.kontrastLabel = tk.Label(self.root, background=self.FOREGROUND_COLOR, text="Kontrast", fg=self.FONT_COLOR)
        self.kontrastLabel.pack(in_=self.kontrastFrame, side=tk.TOP)

        self.kontrastSlider = ttk.Scale(
            self.root,
            from_=-1.0,
            to=1.0,
            orient=tk.VERTICAL,
            style="TScale",
            value=0.0,
            command=self._OnSliderChange,
        )
        self.kontrastSlider.pack(in_=self.kontrastFrame, side=tk.BOTTOM)
        self.kontrastNumber = tk.Label(
            self.root,
            background=self.FOREGROUND_COLOR,
            fg=self.FONT_COLOR,
            text=f"{self.kontrastSlider.get():0.2f}"
        )
        self.kontrastNumber.pack(in_=self.kontrastFrame, side=tk.BOTTOM)

        self._AddSpacing(tk.LEFT, self.imageFrame)

        self.gammaFrame = tk.Frame(self.imageFrame, background=self.FOREGROUND_COLOR)
        self.gammaFrame.pack(side=tk.LEFT)

        self.gammaLabel = tk.Label(self.root, background=self.FOREGROUND_COLOR, text="Gamma", fg=self.FONT_COLOR)
        self.gammaLabel.pack(in_=self.gammaFrame, side=tk.TOP)

        self.gammaSlider = ttk.Scale(
            self.root,
            from_=0,
            to=4.0,
            value=1.0,
            orient=tk.VERTICAL,
            style="TScale",
            command=self._OnSliderChange
        )
        self.gammaSlider.pack(in_=self.gammaFrame, side=tk.BOTTOM)
        self.gammaNumber = tk.Label(
            self.root,
            background=self.FOREGROUND_COLOR,
            fg=self.FONT_COLOR,
            text=f"{self.gammaSlider.get():0.2f}"
        )
        self.gammaNumber.pack(in_=self.gammaFrame, side=tk.BOTTOM)

        self._AddSpacing()

        self.guessButton = tk.Button(
            self.root,
            text="Rozpoznaj znak",
            command=self.GuessSign,
            bg=self.BUTTONS_COLOR,
            fg=self.FONT_COLOR,
            relief=tk.RAISED,
            bd=2,
            font=("Helvetica", 16)
        )
        self.guessButton.pack(in_=self.appFrame)

        self._AddSpacing()

        self.resultLabel = tk.Label(
            self.root,
            text="Oczekiwanie na podanie zdjecia znaku",
            bg=self.BACKGROUND_COLOR,
            fg=self.FONT_COLOR,
            font=("Helvetica", 16)
        )
        self.resultLabel.pack(in_=self.appFrame)

    def LoadPhoto(self) -> None:
        filePath = filedialog.askopenfilename(title="Wybierz plik", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if filePath:
            self.signOryPhoto = self._GetImage(filePath)
            self.SetImagePhotoLabel(self.signOryPhoto)

    def GuessSign(self) -> None:
        if hasattr(self, "photoLabel"):
            predictedIndex = self.model.predict(self.signPhoto)
            predictedSign = self.RoadSignsTransations[predictedIndex]
            self.resultLabel.config(text=f"Znak rozpoznany jako: {predictedSign}")
        else:
            self.resultLabel.config(text="Wybierz najpierw zdjęcie")

    def SetImagePhotoLabel(self, image: Image) -> None:
        self.signPhoto = image
        self.photoLabel = ImageTk.PhotoImage(self.signPhoto)
        self.imageLabel.config(image=self.photoLabel)

    def Run(self) -> None:
        self.root.mainloop()

    @staticmethod
    def _GetImage(iamgePath: Path) -> Image:
        image = Image.open(iamgePath).convert('RGB')
        width, height = image.size

        targetWidth = RoadSignGuesser.SCREEN_WIDTH * 0.6
        targetHeight = RoadSignGuesser.SCREEN_HEIGHT * 0.6
        
        widthPercent = targetWidth / width
        heightPercent = targetHeight / height
        
        maxPercent = min(widthPercent, heightPercent)

        image = image.resize((
                int(width * maxPercent),
                int(height * maxPercent)
            ),
            Image.LANCZOS
        )
        
        return image
    
    def _OnSliderChange(self, sliderValue: float) -> None:
        self.kontrastNumber.config(text=f'{self.kontrastSlider.get():0.2f}')
        self.gammaNumber.config(text=f'{self.gammaSlider.get():0.2f}')
    
        imageArr = np.array(self.signOryPhoto, dtype=np.float32).copy() / 255.
        imageArr *= np.float32(self.gammaSlider.get())
        imageArr += np.float32(self.kontrastSlider.get())
        imageArr = np.clip(imageArr, 0.0, 1.0)
        imageArr = np.uint8(imageArr * 255.0)
        imagePil = Image.fromarray(imageArr)
        self.SetImagePhotoLabel(imagePil)

    def _AddSpacing(self, side: str=tk.TOP, frame: tk.Frame=None):
        if frame is None:
            tk.Label(self.root, text="", bg=self.BACKGROUND_COLOR).pack(in_=self.appFrame, side=side)
        else:
            tk.Label(self.root, text="", bg=self.BACKGROUND_COLOR).pack(in_=frame, side=side)


if __name__ == "__main__":
    app = RoadSignGuesser()
    app.Run()
