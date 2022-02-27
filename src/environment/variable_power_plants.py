from src.environment.generation_models import PowerPlant
from numpy import sign


class LoadFollowingPowerPlant(PowerPlant):
    """
        Base class for load following power plants.
        Implements delays for varying current power output.
        Derived classes must set output_gradient, min_capacity and price_per_mwh.
    """
    def __init__(self, initial_output, max_capacity, step_size, min_capacity, output_gradient,
                 price_per_mwh):
        self.current_output = initial_output
        self.target_output = initial_output
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.step_size = step_size
        self.max_output_gradient = output_gradient * max_capacity * step_size
        self.price_per_mwh = price_per_mwh  # Tonnes of CO2/MWh

    def step(self):
        """Updates power output based on power target, returns current output"""
        if abs(self.current_output - self.target_output) <= self.max_output_gradient:
            self.current_output = self.target_output
        else:
            self.current_output += self.max_output_gradient * sign(self.target_output - self.current_output)
        return self.current_output

    def more(self, delta):
        self.target_output = min(self.current_output + delta * self.max_capacity, self.max_capacity)

    def less(self, delta):
        self.target_output = max(self.current_output - delta * self.max_capacity, self.min_capacity)

    def set_output(self, new_level):
        self.target_output = max(min(self.max_capacity * new_level, self.max_capacity), self.min_capacity)

    def get_costs(self):
        return self.price_per_mwh * self.current_output


class LignitePowerPlant(LoadFollowingPowerPlant):
    def __init__(self, initial_output, max_capacity, step_size):
        super().__init__(initial_output, max_capacity, step_size, 0.004 * max_capacity, 0.04, 0.78)
        

class HardCoalPowerPlant(LoadFollowingPowerPlant):
    def __init__(self, initial_output, max_capacity, step_size):
        super().__init__(initial_output, max_capacity, step_size, 0.004 * max_capacity, 0.03, 1.014)


class GasPowerPlant(LoadFollowingPowerPlant):
    """Based on GuD power plants"""
    def __init__(self, initial_output, max_capacity, step_size):
        super().__init__(initial_output, max_capacity, step_size, 0.002 * max_capacity, 0.06, 0.45)


class NuclearPowerPlant(LoadFollowingPowerPlant):
    def __init__(self, initial_output, max_capacity, step_size):
        # TODO CO2 emissions for uranium mining
        super().__init__(initial_output, max_capacity, step_size, 0.005 * max_capacity, 0.05, 0.0)


class BioMassPowerPlant(LoadFollowingPowerPlant):
    def __init__(self, initial_output, max_capacity, step_size):
        # TODO CO2 emissions for producing biomass
        super().__init__(initial_output, max_capacity, step_size, 0.002 * max_capacity, 0.2, 0.0)









