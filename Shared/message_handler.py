from abc import ABC, abstractmethod

class MessageHandler(ABC):

    @abstractmethod
    def handle_message(self, json_dict: dict):
        pass
