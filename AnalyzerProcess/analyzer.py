import json
import os

from Shared.msg_enum import msg_type


class Analyzer:

    def __init__(self, path: str):

        import analyzer_main
        self.LOGGER = analyzer_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        with open(path, 'r') as f:
            _json_file = json.load(f)

        self._metrics_thresholds_1 = _json_file['_metrics_thresh_1']
        self._metrics_thresholds_2 = _json_file['_metrics_thresh_2']

    def analyze_incoming_metrics(self, metrics1: dict, metrics2: dict, classification_metrics: dict):

        objectives = {
            "MSG_TYPE": str(msg_type.OBJECTIVES_MSG),
            "objs_layer1": [],
            "objs_layer2": []
        }

        for metric, value in metrics1.items():
            if metrics1[metric] < self._metrics_thresholds_1[metric]:
                objectives['objs_layer1'].append(metric)

        for metric, value in metrics2.items():
            if metrics2[metric] < self._metrics_thresholds_2[metric]:
                objectives['objs_layer2'].append(metric)

        self.LOGGER.debug(f'Identified {len(objectives["objs_layer1"])} objective(s) for layer1: '
                          f'[{objectives["objs_layer1"]}]')
        self.LOGGER.debug(f'Identified {len(objectives["objs_layer2"])} objective(s) for layer2: '
                          f'[{objectives["objs_layer2"]}]')

        return objectives
