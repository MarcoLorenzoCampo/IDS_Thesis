from DetectionSystem import DetectionSystem
from KnowledgeBase import KnowledgeBase
from DataProcessor import DataPreprocessingComponent
from Metrics import Metrics
from Plotter import Plotter
from Tuner import Tuner


class DetectionInfrastructure:

    def __init__(self):
        self.kb = KnowledgeBase()
        self.processor = DataPreprocessingComponent()
        self.metrics = Metrics()
        self.plotter = Plotter()
        self.ids = DetectionSystem(self.kb)
        self.hp_tuning = Tuner(self.kb, self.ids)

    def ids(self) -> DetectionSystem:
        return self.ids