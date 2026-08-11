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
    Stores internal memory of visited cells and recent decisions so it can notice loops.
    """

    def __init__(self):
        self.last_action = None
        self.last_percept = {}
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
        # Update internal state before choosing a rule-based action.
        self.last_percept = dict(percept)

        if self.last_action in {"Up", "Down", "Left", "Right"}:
            self.position = self._step(self.position, self.last_action)
            self.visited.add(self.position)

        if percept.get("food_here"):
            self.last_action = "EAT"
            return "EAT"

        if percept.get("wall_ahead"):
            left_dir = self._turn(self.direction, "Left")
            right_dir = self._turn(self.direction, "Right")

            # IF wall_ahead AND left_is_unvisited THEN turn_left
            if self._step(self.position, left_dir) not in self.visited:
                self.direction = left_dir
                self.last_action = left_dir
                return left_dir

            # IF wall_ahead AND left_is_visited AND right_is_unvisited THEN turn_right
            if self._step(self.position, right_dir) not in self.visited:
                self.direction = right_dir
                self.last_action = right_dir
                return right_dir

            # IF wall_ahead AND both sides are visited -> take a fallback turn to break the loop
            fallback = self._turn(self.direction, "Left")
            self.direction = fallback
            self.last_action = fallback
            return fallback

        # ELSE continue moving in the current direction.
        self.direction = self.direction
        self.last_action = self.direction
        return self.direction