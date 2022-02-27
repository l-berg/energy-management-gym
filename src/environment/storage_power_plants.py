from src.environment.generation_models import PowerPlant


class StoragePowerPlant(PowerPlant):
    def __init__(self, initial_output, initial_energy_level, step_size, output_increment, max_output_capacity,
                 max_charge_capacity, storage_capacity, price_per_mwh_charge, price_per_mwh_output):
        self.current_output = initial_output
        self.energy_level = initial_energy_level
        self.step_size = step_size
        self.output_increment = output_increment
        self.max_output_capacity = max_output_capacity
        self.max_charge_capacity = max_charge_capacity
        self.storage_capacity = storage_capacity
        self.price_per_mwh_output = price_per_mwh_output
        self.price_per_mwh_charge = price_per_mwh_charge

    def step(self):
        if self.current_output + self.energy_level < self.storage_capacity:
            self.energy_level = self.storage_capacity
            self.current_output = 0.0
            return self.current_output
        elif self.current_output + self.energy_level >= 0:
            production = self.energy_level
            self.energy_level = 0
            return production
        self.energy_level += self.current_output
        return self.current_output

    def more(self):
        self.current_output = min(self.current_output + self.output_increment *
                                  (self.max_output_capacity - self.max_charge_capacity), self.max_output_capacity)

    def less(self):
        self.current_output = max(self.current_output - self.output_increment *
                                  (self.max_output_capacity - self.max_charge_capacity), self.max_charge_capacity)

    def get_costs(self):
        if self.current_output >= 0:
            return self.price_per_mwh_output * self.current_output
        else:
            return self.price_per_mwh_charge * self.current_output




