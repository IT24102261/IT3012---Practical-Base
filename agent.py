import random
from collections import deque
import heapq


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
    Stores internal memory of visited cells and recent decisions
    so it can notice loops.
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

            # IF wall_ahead AND left_is_visited AND right_is_unvisited
            # THEN turn_right
            if self._step(self.position, right_dir) not in self.visited:
                self.direction = right_dir
                self.last_action = right_dir
                return right_dir

            # Fallback turn to break the loop
            fallback = self._turn(self.direction, "Left")
            self.direction = fallback
            self.last_action = fallback
            return fallback

        # Continue moving in the current direction.
        self.last_action = self.direction
        return self.direction


class SearchAgent:
    """
    Goal-Based Search Agent.

    Supports:
        - Breadth-First Search (BFS)
        - Depth-First Search (DFS)
        - Uniform-Cost Search (UCS)

    The agent first creates a complete plan to a food location,
    then executes that plan one action at a time.
    """

    def __init__(self):
        # Step 1.3 requirement
        self.plan = []

        # Default search algorithm
        self.active_algo = "BFS"

    def _get_neighbors(self, position, walls, grid_size):
        """
        Return valid neighboring states and the action required
        to reach each state.

        Returns:
            [(next_position, action), ...]
        """

        x, y = position
        width, height = grid_size

        possible_moves = [
            ("Up", (x, y + 1)),
            ("Down", (x, y - 1)),
            ("Left", (x - 1, y)),
            ("Right", (x + 1, y)),
        ]

        neighbors = []

        for action, next_position in possible_moves:
            nx, ny = next_position

            # Check grid boundaries
            inside_grid = (
                0 <= nx < width
                and 0 <= ny < height
            )

            # Check walls
            not_wall = next_position not in walls

            if inside_grid and not_wall:
                neighbors.append((next_position, action))

        return neighbors

    # ---------------------------------------------------------
    # BFS
    # ---------------------------------------------------------

    def bfs_search(self, start, goal, walls, grid_size):
        """
        Breadth-First Search.

        Uses:
            FIFO queue -> deque.popleft()

        BFS explores the shallowest nodes first.
        """

        if start == goal:
            return []

        # FIFO frontier
        frontier = deque()

        # Store:
        # (current_state, path_to_current_state)
        frontier.append((start, []))

        # Reached set prevents repeated states
        reached = {start}

        while frontier:
            current, path = frontier.popleft()

            for next_state, action in self._get_neighbors(
                current,
                walls,
                grid_size
            ):
                # Ignore already visited states
                if next_state in reached:
                    continue

                new_path = path + [action]

                # Goal found
                if next_state == goal:
                    return new_path

                # Mark as reached
                reached.add(next_state)

                # Add to FIFO queue
                frontier.append((next_state, new_path))

        # No path found
        return None

    # ---------------------------------------------------------
    # DFS
    # ---------------------------------------------------------

    def dfs_search(self, start, goal, walls, grid_size):
        """
        Depth-First Search.

        Uses:
            LIFO stack -> list.pop()

        DFS explores the deepest available path first.
        """

        if start == goal:
            return []

        # LIFO stack
        frontier = [(start, [])]

        # Reached set prevents loops
        reached = {start}

        while frontier:
            current, path = frontier.pop()

            for next_state, action in self._get_neighbors(
                current,
                walls,
                grid_size
            ):
                # Ignore already visited states
                if next_state in reached:
                    continue

                new_path = path + [action]

                # Goal found
                if next_state == goal:
                    return new_path

                # Mark as reached
                reached.add(next_state)

                # Add to stack
                frontier.append((next_state, new_path))

        # No path found
        return None

    # ---------------------------------------------------------
    # UCS
    # ---------------------------------------------------------

    def ucs_search(self, start, goal, walls, grid_size):
        """
        Uniform-Cost Search.

        Uses:
            Priority queue -> heapq

        The priority is the total path cost g(n).

        In this grid, every movement has a cost of 1.
        """

        if start == goal:
            return []

        # Priority queue:
        # (total_cost, state, path)
        frontier = []

        heapq.heappush(
            frontier,
            (0, start, [])
        )

        # Store the cheapest cost found for each state
        reached = {
            start: 0
        }

        while frontier:
            cost, current, path = heapq.heappop(frontier)

            # Goal found
            if current == goal:
                return path

            for next_state, action in self._get_neighbors(
                current,
                walls,
                grid_size
            ):
                # Every movement costs 1
                new_cost = cost + 1

                new_path = path + [action]

                # Add the state if:
                # 1. We have not reached it before
                # OR
                # 2. We found a cheaper path
                if (
                    next_state not in reached
                    or new_cost < reached[next_state]
                ):
                    reached[next_state] = new_cost

                    heapq.heappush(
                        frontier,
                        (
                            new_cost,
                            next_state,
                            new_path
                        )
                    )

        # No path found
        return None

    # ---------------------------------------------------------
    # Sense and Act
    # ---------------------------------------------------------

    def sense_and_act(self, percept):
        """
        Create a complete plan when the current plan is empty,
        then execute the plan one action at a time.
        """

        # Only create a new plan when there is no existing plan.
        if not self.plan:

            # Current agent position
            start = tuple(percept.get("agent_pos", (0, 0)))

            # Get all food locations
            all_food = percept.get("all_food", [])

            # If there is no food left
            if not all_food:
                return "EAT"

            # Find the closest food using Manhattan distance.
            target = min(
                all_food,
                key=lambda food: (
                    abs(food[0] - start[0])
                    + abs(food[1] - start[1])
                )
            )

            # Get world information
            walls = set(percept.get("walls", []))
            grid_size = percept.get("grid_size", (10, 10))

            # Select the requested search algorithm
            if self.active_algo == "BFS":

                self.plan = self.bfs_search(
                    start,
                    target,
                    walls,
                    grid_size
                )

            elif self.active_algo == "DFS":

                self.plan = self.dfs_search(
                    start,
                    target,
                    walls,
                    grid_size
                )

            elif self.active_algo == "UCS":

                self.plan = self.ucs_search(
                    start,
                    target,
                    walls,
                    grid_size
                )

            else:
                raise ValueError(
                    f"Unknown search algorithm: {self.active_algo}"
                )

            # If no path exists
            if self.plan is None:
                return "Up"

        # Execute the first action in the plan
        return self.plan.pop(0)