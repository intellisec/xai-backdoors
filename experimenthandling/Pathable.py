import pathlib
import shutil


class Pathable:
    """
    A simple wrapper around the directory structure.
    """
    def __init__(self,path):
        if type(path) is str:
            path = pathlib.Path(path)
        self.path = path

    def delete(self):
        shutil.rmtree(self.path)

    def create(self):
        self.path.mkdir(exist_ok=True)

    def exists(self) -> bool:
        return self.path.exists()

    def must_exist(self):
        if not self.path.exists():
            raise Exception(f'Path {self.path} does not exist!')