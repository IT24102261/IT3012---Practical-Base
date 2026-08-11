import random


class SimpleReflexAgent:
    """
    A Simple Reflex Agent.
    Makes decisions using only the current percept.
    No memory is stored.
    """

    def sense_and_act(self, percept):
        # IF food is in front of the agent, take it.
        if percept.get("food_here"):
            return "EAT"

        # IF a wall is directly ahead, turn left.
        if percept.get("wall_ahead"):
            return "Left"

        # ELSE move forward.
        return "Up"


class ModelBasedAgent:
    """
    A Model-Based Agent.
    Keeps internal memory of visited cells and an internal direction model so it can avoid looping.
    """

    def __init__(self):
        self.last_action = None
        self.position = (0, 0)
        self.direction = "Up"
        self.visited = {(0, 0)}

    def _turn(self, current, turn):
        order = {
            "Up": {"Left": "Left", "Right": "Right"},
            "Down": {"Left": "Right", "Right": "Left"},
            "Left": {"Left": "Down", "Right": "Up"},
            "Right": {"Left": "Up", "Right": "Down"},
        }
        return order[current][turn]

    def _step(self, pos, action):
        x, y = pos
        if action == "Up":
            return (x, y + 1)
        if action == "Down":
            return (x, y - 1)
        if action == "Left":
            return (x - 1, y)
        if action == "Right":
            return (x + 1, y)
        return pos

    def sense_and_act(self, percept):
        if self.last_action in {"Up", "Down", "Left", "Right"}:
            self.position = self._step(self.position, self.last_action)
            self.visited.add(self.position)

        if percept.get("food_here"):
            self.last_action = "EAT"
            return "EAT"

        if percept.get("wall_ahead"):
            for turn in ["Left", "Right"]:
                candidate_dir = self._turn(self.direction, turn)
                candidate_pos = self._step(self.position, candidate_dir)
                if candidate_pos not in self.visited:
                    self.direction = candidate_dir
                    self.last_action = candidate_dir
                    return candidate_dir

            fallback = self._turn(self.direction, "Left")
            self.direction = fallback
            self.last_action = fallback
            return fallback

        self.last_action = self.direction
        return self.direction