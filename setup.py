import sys
from pathlib import Path

from utils import *
PythonVersionError.validate_python()


def create_virtual_env(venv_name = "venv"):
    try:
        import venv
    except ImportError:
        print('Module "venv" not found')
        sys.exit(1)

    venv_path = Path.cwd() / venv_name

    if venv_path.exists():
        print(f"Virtual environment '{venv_name}' already exists.")
        sys.exit(1)

    venv.create(venv_path, with_pip=True)

    print(f'Virtual environment "{venv_name}" has been created.')


if __name__ == "__main__":
    create_virtual_env("ImageVenv")
