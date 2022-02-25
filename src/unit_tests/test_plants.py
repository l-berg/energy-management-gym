import src.environment.generation_models as gm

import pytest

def test_inst_log_plant():
    starting_output = 45
    max_capacity = 100
    plant = gm.InstLogPlant(current_output=starting_output, max_capacity=max_capacity)
    assert plant.step() == starting_output

    inc = 5
    for _ in range(inc):
        plant.more()

    expected_generation = starting_output * (1+plant.CHANGE_FACTOR)**inc
    assert plant.step() == pytest.approx(expected_generation)

    dec = 7
    for _ in range(dec):
        plant.less()

    expected_generation *= (1-plant.CHANGE_FACTOR)**dec
    plant.step()
    assert plant.step() == pytest.approx(expected_generation)

    for _ in range(100):
        plant.more()
    assert plant.step() == max_capacity

    for _ in range(100):
        plant.less()
    assert plant.step() < 0.1*max_capacity
