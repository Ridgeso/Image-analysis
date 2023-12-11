
class ImageProcessingError(Exception):
    pass


class PythonVersionError(ValueError):
    def __init__(self) -> None:
        super().__init__("This Project requires Python >= 3.3")

    @staticmethod
    def validate_python() -> None:
        from sys import version_info
        if version_info < (3, 3):
            raise PythonVersionError()
