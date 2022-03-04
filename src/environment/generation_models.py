class PowerPlant:
    """Parent class for all power generation models"""
    def __init__(self, current_output, max_capacity):
        self.current_output = current_output
        self.max_capacity = max_capacity

    def step(self):
        """Returns the amount of power generated"""
        pass

    def more(self, delta):
        """Signal to increase generation by fraction of max_capacity"""
        pass

    def less(self, delta):
        """Signal to decrease generation by fraction of max_capacity"""
        pass

    def change_output(self, delta):
        """Signal to change generation by fraction of max_capacity"""
        if delta < 0:
            self.less(-delta)
        elif delta > 0:
            self.more(delta)

    def set_output(self, new_level):
        """Signal to set output to desired fraction of max_capacity"""
        pass

    def get_costs(self):
        """Returns the cost of operating the power plant at the current level"""
        pass


class InstLogPlant(PowerPlant):
    """Changes take effect instantly"""
    CHANGE_FACTOR = 0.1

    def step(self):
        return self.current_output

    def more(self, delta):
        self.current_output = min(self.current_output * (1+self.CHANGE_FACTOR), self.max_capacity)

    def less(self, delta):
        self.current_output = max(self.current_output * (1-self.CHANGE_FACTOR), 0)

    def set_output(self, new_level):
        self.current_output = new_level * self.max_capacity

    def get_costs(self):
        return self.current_output

