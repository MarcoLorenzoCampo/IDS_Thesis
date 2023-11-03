class Metrics:
    def __init__(self):
        self._metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    def update(self, tag, value):
        self._metrics[tag] += value

    def get(self, tag):
        return self._metrics[tag]