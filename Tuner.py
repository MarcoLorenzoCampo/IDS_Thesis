from KnowledgeBase import KnowledgeBase
from DetectionSystem import DetectionSystem


class Tuner:
    """
    This class is used to perform hyperparameter tuning on the two classifiers
    """

    def __init__(self, kb: KnowledgeBase, ids: DetectionSystem):

        # do not modify the values, passed by reference
        # validation sets
        self.x_validate_1 = kb.x_validate_l1
        self.x_validate_2 = kb.x_validate_l2
        self.y_validate_l1 = kb.y_validate_l1
        self.y_validate_l2 = kb.y_validate_l2

        # train sets
        self.x_train_1 = kb.x_train_l1
        self.x_train_2 = kb.x_train_l2
        self.y_train_1 = kb.y_train_l1
        self.y_train_1 = kb.y_train_l1

        # classifiers
        self.layer1 = ids.layer1
        self.layer2 = ids.layer2

    def tune(self, objs: set):
        """
        This function handles the hyperparameter tuning for both of the classifiers
        :param objs: objectives to reach
        :return:
        """

    def hp_iterator(self, hp1: set, hp2: set):
        return