import pandas as pd

class PercentileDetection:

    def __init__(self, percentile=0.9):
        self.percentile = percentile

    def fit(self, x, y=None):
        self.thresholds = [
            pd.Series(x[:,i]).quantile(self.percentile)
            for i in range(x.shape[1])
        ]

    def predict(self, x, y=None):
        return (x > self.thresholds).max(axis=1)

    def fit_predict(self, x, y=None):
        self.fit(x)
        return self.predict(x)