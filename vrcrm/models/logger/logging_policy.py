from abc import abstractmethod, ABC


class LoggingPolicy(ABC):
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.model = None

    @abstractmethod
    def generateLog(self, dataset):
        return None, None, None
