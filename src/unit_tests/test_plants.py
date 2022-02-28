import src.environment.generation_models as gm
import src.environment.variable_power_plants as vp
import src.environment.storage_power_plants as sp

import pytest


def test_inst_log_plant():
    starting_output = 45
    max_capacity = 100
    delta = 0.1
    plant = gm.InstLogPlant(current_output=starting_output, max_capacity=max_capacity)
    assert plant.step() == starting_output

    inc = 5
    for _ in range(inc):
        plant.more(delta)

    expected_generation = starting_output * (1+delta)**inc
    assert plant.step() == pytest.approx(expected_generation)

    dec = 7
    for _ in range(dec):
        plant.less(delta)

    expected_generation *= (1-delta)**dec
    plant.step()
    assert plant.step() == pytest.approx(expected_generation)

    for _ in range(100):
        plant.more(delta)
    assert plant.step() == max_capacity

    for _ in range(100):
        plant.less(delta)
    assert plant.step() < 0.1*max_capacity


def test_load_following_plant():
    initial_output = 500
    max_capacity = 1000
    delta = 0.1
    step_size = 1
    plant = vp.LignitePowerPlant(initial_output, max_capacity, step_size, 'single')

    # test initial production
    assert plant.step() == initial_output

    # test increasing output and output delay
    output_gradient = plant.max_output_gradient
    plant.more(delta)
    new_output = plant.step()
    assert new_output == pytest.approx(initial_output + output_gradient)

    # test output target
    for _ in range(5):
        new_output = plant.step()
    assert new_output == pytest.approx(initial_output + delta * max_capacity)

    # test decreasing output
    plant.less(delta)
    old_output = new_output
    new_output = plant.step()
    assert new_output == pytest.approx(old_output - output_gradient)

    # test output capacity
    for _ in range(100):
        plant.more(delta)
        plant.step()
    assert plant.step() == max_capacity

    # test minimum output
    for _ in range(100):
        plant.less(delta)
        plant.step()
    assert plant.step() == plant.min_capacity

    # test setting output level
    plant.set_output(0.9)
    assert plant.step() == pytest.approx(plant.min_capacity + output_gradient)
    for _ in range(50):
        plant.step()
    assert plant.step() == pytest.approx(0.9 * max_capacity)


def test_storage_plant():
    initial_output = 100
    max_output = 1000
    max_charge = 1000
    max_storage = 10000
    initial_storage = 100
    delta = 0.1
    efficiency = 0.8
    plant = sp.StoragePowerPlant(initial_output, initial_storage, 1, max_output, max_charge,
                                 max_storage, efficiency)

    # test initial output
    assert plant.step() == efficiency * initial_output

    # test empty storage
    assert plant.step() == 0

    # test charging storage
    plant.less(delta)
    new_output = plant.step()
    assert new_output == pytest.approx(delta * -max_charge)
    assert plant.energy_level == pytest.approx(new_output)

    # test fully charging
    for _ in range(10):
        plant.less(delta)
    for _ in range(10):
        plant.step()
    assert plant.energy_level == -max_storage
    assert plant.step() == 0

    # test total output from full to empty
    for _ in range(20):
        plant.more(delta)
    total_output = 0
    for _ in range(30):
        total_output += plant.step()
        print(total_output)
    assert total_output == max_storage * efficiency









