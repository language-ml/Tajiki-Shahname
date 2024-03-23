class BestFinder:
    def __init__(self, higher_better=True):
        self.best_value = None
        self.higher_better = higher_better
        
    def _compare(self, new_value):
        if self.best_value is None:
            return True
        
        if self.higher_better:
            return new_value > self.best_value
        else:
            return new_value < self.best_value
        
    def is_better(self, new_value):
        compare_reuslt = self._compare(new_value)
        if compare_reuslt:
            self.best_value = new_value
        return compare_reuslt
