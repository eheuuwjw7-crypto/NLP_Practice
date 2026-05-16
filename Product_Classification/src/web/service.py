class TitleService:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, title):
        result = self.predictor.predict(title)
        return result["label"]
