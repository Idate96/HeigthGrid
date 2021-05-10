import unittest
import numpy as np 
from heightgrid.heightgrid import Goal, Agent, Ramp

class TestObjects(unittest.TestCase):

    def test_goal_init(self):
        goal = Goal()
        self.assertEqual('goal', goal.type)
        self.assertEqual('green', goal.color)
        self.assertEqual(True, goal.can_overlap())

    def test_ramp_init(self):
        ramp = Ramp()
        self.assertEqual(True, ramp.orientable)
    
    def test_agent_init(self):
        agent = Agent()
        self.assertEqual(True, agent.orientable)
        self.assertEqual('agent', agent.type)


if __name__ == '__main__':
    unittest.main()