from src.environment.generation_models import PowerPlant
from numpy import sign


class LoadFollowingPowerPlant(PowerPlant):
    """
        Base class for load following power plants.
        Implements delay for varying current power output.
        Derived classes must set output_gradient, min_capacity and price_per_mwh.
    """
    def __init__(self, initial_output, max_capacity, step_size, min_capacity, output_gradient,
                 price_per_mwh, mode, shutdown_penalty=1):
        self.current_output = initial_output
        self.target_output = initial_output
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.step_size = step_size
        self.max_output_gradient = output_gradient * max_capacity * step_size
        self.price_per_mwh = price_per_mwh  # Tonnes of CO2/MWh
        assert mode == 'single' or mode == 'group'
        self.mode = mode
        self.shutdown_penalty = shutdown_penalty

    def step(self):
        """Updates power output based on power target, returns current output"""
        # calculate portion of time spend above and below min capacity
        part_below_min = 0
        part_above_min = 0
        if self.current_output <= self.min_capacity and self.target_output <= self.min_capacity:
            part_below_min = 1
        elif self.current_output <= self.min_capacity <= self.target_output:
            part_below_min = min((self.min_capacity - self.current_output) /
                                 (self.max_output_gradient * self.shutdown_penalty), 1)
            part_above_min = 1 - part_below_min
        elif self.current_output >= self.min_capacity and self.target_output >= self.min_capacity:
            part_above_min = 1
        elif self.current_output >= self.min_capacity >= self.target_output:
            part_above_min = min((self.current_output - self.min_capacity) / self.max_output_gradient, 1)
            part_below_min = 1 - part_above_min

        # calculate actual output by combining change with and without shutdown penalty
        desired_output = self.current_output + sign(self.target_output - self.current_output) * \
                         self.max_output_gradient * (part_above_min + part_below_min * self.shutdown_penalty)

        if abs(desired_output - self.current_output) > abs(self.target_output - self.current_output):
            desired_output = self.target_output

        self.current_output = desired_output
        return self.current_output

    def more(self, delta):
        self.target_output = min(self.current_output + delta * self.max_capacity, self.max_capacity)

    def less(self, delta):
        if self.mode == 'single':
            self.target_output = max(self.current_output - delta * self.max_capacity, self.min_capacity)
        else:
            self.target_output = max(self.current_output - delta * self.max_capacity, 0.0)

    def set_output(self, new_level):
        if self.mode == 'single':
            self.target_output = max(min(self.max_capacity * new_level, self.max_capacity), self.min_capacity)
        else:
            self.target_output = max(min(self.max_capacity * new_level, self.max_capacity), 0)

    def get_costs(self):
        return self.price_per_mwh * self.current_output


class LignitePowerPlant(LoadFollowingPowerPlant):
    def __init__(self, initial_output, max_capacity, step_size, mode):
        super().__init__(initial_output, max_capacity, step_size, 0.4 * max_capacity, 0.04, 0.78, mode, 0.1)


class HardCoalPowerPlant(LoadFollowingPowerPlant):
    def __init__(self, initial_output, max_capacity, step_size, mode):
        super().__init__(initial_output, max_capacity, step_size, 0.4 * max_capacity, 0.03, 1.014, mode, 0.08)


class GasPowerPlant(LoadFollowingPowerPlant):
    """Based on GuD power plants"""
    def __init__(self, initial_output, max_capacity, step_size, mode):
        super().__init__(initial_output, max_capacity, step_size, 0.2 * max_capacity, 0.06, 0.45, mode, 0.9)


class NuclearPowerPlant(LoadFollowingPowerPlant):
    def __init__(self, initial_output, max_capacity, step_size, mode):
        super().__init__(initial_output, max_capacity, step_size, 0.5 * max_capacity, 0.05, 0.0, mode, 0.5)


class BioMassPowerPlant(LoadFollowingPowerPlant):
    def __init__(self, initial_output, max_capacity, step_size, mode):
        super().__init__(initial_output, max_capacity, step_size, 0.2 * max_capacity, 0.2, 0.0, mode, 0.9)









