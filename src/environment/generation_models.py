
class PowerPlant:
    """Parent class for all power generation models"""
    def __init__(self, current_output, max_capacity):
        self.current_output = current_output
        self.max_capacity = max_capacity

    def step(self):
        """Returns the amount of power generated"""
        pass

    def more(self):
        """Signal to increase generation"""
        pass

    def less(self):
        """Signal to decrease generation"""
        pass

class InstLogPlant(PowerPlant):
    """Changes take effect instantly"""
    CHANGE_FACTOR = 0.1

    def step(self):
        return self.current_output

    def more(self):
        self.current_output = min(self.current_output * (1+self.CHANGE_FACTOR), self.max_capacity)

    def less(self):
        self.current_output = max(self.current_output * (1-self.CHANGE_FACTOR), 0)
