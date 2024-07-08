class SorterSelector:
    def __init__(self, property_name: str):
        self.property_name = property_name

    def predicate(self):
        return lambda x: x[self.property_name]


class RewardSorterSelector(SorterSelector):
    def __init__(self):
        self.property_name = "reward"
