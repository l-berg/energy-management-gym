from src.environment.generation_models import PowerPlant


class StoragePowerPlant(PowerPlant):
    """
        Base class for storage power plants.
        Does not have any output delays, but can only generate power while internal storage is not empty.
        Generating power from internal storage leads to some energy conversion losses.
    """
    def __init__(self, initial_output, initial_energy_level, step_size, max_output_capacity,
                 max_charge_capacity, storage_capacity, efficiency):
        self.current_output = initial_output
        self.energy_level = -initial_energy_level
        self.step_size = step_size
        self.max_output_capacity = max_output_capacity
        self.max_charge_capacity = -max_charge_capacity
        self.storage_capacity = -storage_capacity
        self.efficiency = efficiency
        self.price_per_mwh = 0  # for reward calculation

    def step(self):
        """
            Updates internal energy storage and returns current output.
            Negative output: power is stored.
            Positive output: power is generated.
        """
        # storage full
        if self.current_output + self.energy_level < self.storage_capacity:
            self.energy_level = self.storage_capacity
            self.current_output = 0.0
            return self.current_output
        # storage empty
        elif self.current_output + self.energy_level >= 0:
            production = -self.energy_level
            self.energy_level = 0
            print("empty", production)
            return self.efficiency * production
        self.energy_level += self.current_output

        if self.current_output < 0:
            return self.current_output
        else:
            return self.efficiency * self.current_output

    def more(self, delta):
        self.current_output = min(self.current_output + delta *
                                  (self.max_output_capacity - self.max_charge_capacity), self.max_output_capacity)

    def less(self, delta):
        self.current_output = max(self.current_output - delta *
                                  (self.max_output_capacity - self.max_charge_capacity), self.max_charge_capacity)

    def set_output(self, new_level):
        self.current_output = max(min(new_level * self.max_output_capacity, self.max_output_capacity),
                                  self.max_charge_capacity)

    def get_costs(self):
        return 0




