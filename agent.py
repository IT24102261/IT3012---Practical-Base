# agent.py

from collections import deque
import heapq


class SearchAgent:

    def __init__(self):

        # Required by Practical 03
        self.plan = []

        # Change between BFS, DFS and UCS
        self.active_algo = "ASTAR"


    # ---------------------------------------------------------
    # Heuristic Functions
    # ---------------------------------------------------------

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return ((pos[0] - goal[0]) ** 2 +
                (pos[1] - goal[1]) ** 2) ** 0.5

    # ---------------------------------------------------------
    # Get valid neighbouring cells
    # ---------------------------------------------------------

    def get_neighbors(self, state, grid_size, walls):

        x, y = state

        width, height = grid_size

        possible_moves = [
            ((x, y + 1), "Up"),
            ((x, y - 1), "Down"),
            ((x - 1, y), "Left"),
            ((x + 1, y), "Right")
        ]

        neighbors = []

        for next_state, action in possible_moves:

            nx, ny = next_state

            # Check grid boundaries
            if nx < 0 or nx >= width:
                continue

            if ny < 0 or ny >= height:
                continue

            # Check wall
            if next_state in walls:
                continue

            neighbors.append((next_state, action))

        return neighbors

    # ---------------------------------------------------------
    # BFS
    # ---------------------------------------------------------

    def bfs_search(self, start, goal, grid_size, walls):

        frontier = deque()

        frontier.append(
            (start, [])
        )

        reached = {start}

        while frontier:

            state, path = frontier.popleft()

            # Goal found
            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state,
                grid_size,
                walls
            ):

                if next_state not in reached:

                    reached.add(next_state)

                    frontier.append(
                        (
                            next_state,
                            path + [action]
                        )
                    )

        return []

    # ---------------------------------------------------------
    # DFS
    # ---------------------------------------------------------

    def dfs_search(self, start, goal, grid_size, walls):

        frontier = []

        frontier.append(
            (start, [])
        )

        reached = {start}

        while frontier:

            state, path = frontier.pop()

            # Goal found
            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state,
                grid_size,
                walls
            ):

                if next_state not in reached:

                    reached.add(next_state)

                    frontier.append(
                        (
                            next_state,
                            path + [action]
                        )
                    )

        return []

    # ---------------------------------------------------------
    # UCS
    # ---------------------------------------------------------

    def ucs_search(self, start, goal, grid_size, walls):

        frontier = []

        # (cost, counter, state, path)
        counter = 0

        heapq.heappush(
            frontier,
            (0, counter, start, [])
        )

        reached = {
            start: 0
        }

        while frontier:

            cost, _, state, path = heapq.heappop(
                frontier
            )

            # Goal found
            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state,
                grid_size,
                walls
            ):

                new_cost = cost + 1

                if (
                    next_state not in reached
                    or new_cost < reached[next_state]
                ):

                    reached[next_state] = new_cost

                    counter += 1

                    heapq.heappush(
                        frontier,
                        (
                            new_cost,
                            counter,
                            next_state,
                            path + [action]
                        )
                    )

        return []
            

    # ---------------------------------------------------------
    # A* Search
    # ---------------------------------------------------------

    def astar_search(
        self,
        start_pos,
        goal_pos,
        walls,
        grid_size,
        heuristic_type="manhattan"
    ):

        frontier = []

        # (f_cost, g_cost, current_pos, path_taken)

        start_h = self.manhattan_distance(
            start_pos,
            goal_pos
        )

        heapq.heappush(
            frontier,
            (
                start_h,
                0,
                start_pos,
                []
            )
        )

        reached_states = set()

        while frontier:

            f_cost, g_cost, current_pos, path_taken = heapq.heappop(
                frontier
            )

            # Goal found
            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue

            reached_states.add(current_pos)

            for next_pos, action in self.get_neighbors(
                current_pos,
                grid_size,
                walls
            ):

                if next_pos in reached_states:
                    continue

                # g(n)
                new_g = g_cost + 1

                # h(n)
                if heuristic_type == "euclidean":

                    new_h = self.euclidean_distance(
                        next_pos,
                        goal_pos
                    )

                else:

                    new_h = self.manhattan_distance(
                        next_pos,
                        goal_pos
                    )

                # f(n) = g(n) + h(n)
                new_f = new_g + new_h

                heapq.heappush(
                    frontier,
                    (
                        new_f,
                        new_g,
                        next_pos,
                        path_taken + [action]
                    )
                )

        return []

   
    # ---------------------------------------------------------
    # Find closest food
    # ---------------------------------------------------------

    def find_closest_food(
        self,
        start,
        food_positions,
        grid_size,
        walls
    ):

        best_path = None
        best_food = None

        for food in food_positions:

            if self.active_algo == "BFS":

                path = self.bfs_search(
                    start,
                    food,
                    grid_size,
                    walls
                )

            elif self.active_algo == "DFS":

                path = self.dfs_search(
                    start,
                    food,
                    grid_size,
                    walls
                )

            elif self.active_algo == "UCS":

                path = self.ucs_search(
                    start,
                    food,
                    grid_size,
                    walls
                )
            elif self.active_algo == "ASTAR":

                path = self.astar_search(
                    start,
                    food,
                    walls,
                    grid_size,
                    "manhattan"
                )
            else:

                path = self.bfs_search(
                    start,
                    food,
                    grid_size,
                    walls
                )

            # Ignore unreachable food
            if path == [] and start != food:
                continue

            if best_path is None or len(path) < len(best_path):

                best_path = path
                best_food = food

        if best_path is None:

            return []

        return best_path

    # ---------------------------------------------------------
    # Sense and Act
    # ---------------------------------------------------------

    def sense_and_act(self, percept, current_position):

        # If there is no current plan,
        # calculate a new plan.
        if not self.plan:

            grid_size = percept["grid_size"]

            walls = set(
                tuple(wall)
                for wall in percept["walls"]
            )

            all_food = [
                tuple(food)
                for food in percept["all_food"]
            ]

            # No food remaining
            if not all_food:
                return "Up"

            # Create a plan
            self.plan = self.find_closest_food(
                current_position,
                all_food,
                grid_size,
                walls
            )

        # Execute next action
        if self.plan:

            return self.plan.pop(0)

        return "Up"

if __name__ == "__main__":

    agent = SearchAgent()

    start = (0, 0)
    goal = (3, 4)

    print("Manhattan:", agent.manhattan_distance(start, goal))
    print("Euclidean:", agent.euclidean_distance(start, goal))

    walls = set()
    grid_size = (5, 5)

    path = agent.astar_search(
        (0, 0),
        (3, 4),
        walls,
        grid_size
    )

    print("A* path:", path)