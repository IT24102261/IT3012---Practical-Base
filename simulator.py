# simulator.py
from visual_grid_game import VisualGridHuntGame
from agent import ModelBasedAgent


def run_grid_hunt():
    env = VisualGridHuntGame(
        width=6,
        height=6,
        num_food=2,
        num_opponents=0,
        custom_walls={(1, 0), (1, 1), (1, 2), (2, 2), (3, 2), (4, 2)},
    )
    agent = ModelBasedAgent()

    print("=== Model-Based Agent Simulation ===")
    for step in range(30):
        percept = env.get_percept()
        action = agent.sense_and_act(percept)
        print(f"Step {step + 1}: pos={env.agent_pos}, facing={env.facing}, percept={percept}, action={action}")
        env.execute_action(action)

        if env.is_done():
            break

    print(f"\nFinal Score: {env.score} | Steps: {env.steps}")


if __name__ == "__main__":
    run_grid_hunt()