class Metrics:
    def __init__(self):
        self._metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        if isinstance(value, dict) and all(key in value for key in ('tp', 'tn', 'fp', 'fn')):
            self._metrics = value
        else:
            raise ValueError("Invalid value for metrics.")
