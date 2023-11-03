from DetectionSystem import DetectionSystem
from KnowledgeBase import KnowledgeBase
from DataPreprocessingComponent import DataPreprocessingComponent
from Metrics import Metrics
from Plotter import Plotter


class DetectionInfrastructure:

    def __init__(self):
        self.kb = KnowledgeBase()
        self.processor = DataPreprocessingComponent()
        self.metrics = Metrics()
        self.plotter = Plotter()
        self.ids = DetectionSystem(self.kb)

    def ids(self) -> DetectionSystem:
        return self.ids
