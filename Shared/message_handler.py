from abc import ABC, abstractmethod

class MetricsMsgHandler(ABC):

    @abstractmethod
    def handle_metrics_msg(self, json_dict: dict):
        pass

class SimpleMsgHandler(ABC):

    @abstractmethod
    def handle_models_update_msg(self, json_dict: dict):
        pass

class FullMsgHandler(SimpleMsgHandler):

    @abstractmethod
    def handle_multiple_updates_msg(self, json_dict: dict):
        pass

    @abstractmethod
    def handle_objs_msg(self, json_dict: dict):
        pass

