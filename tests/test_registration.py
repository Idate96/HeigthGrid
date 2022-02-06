"""Test wheter the heightgrid env is registered in gym."""

import gym

from heightgrid.envs_v2 import HeightGridEnvV2

def test_registration():
    """Test wheter the heightgrid env is registered in gym."""
    assert "HeightGrid-v2" in gym.envs.registry.all()

if __name__ == "__main__":
    test_registration()

