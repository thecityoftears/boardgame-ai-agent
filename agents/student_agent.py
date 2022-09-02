# Student agent: Add your own agent here

import math
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.start_pos = None
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # dummy return

        if self.start_pos is None:
            self.start_pos = my_pos
        barriers = self.find_barriers_to_build(chess_board, self.start_pos)
        player_grid = self.board_parser(chess_board, my_pos, adv_pos)
        closest = self.find_closest_barrier(player_grid, barriers)
        barrier_coord = (closest[0][0], closest[0][1])
        barrier_grid = self.board_parser(chess_board, barrier_coord, adv_pos)
        m = len(chess_board)

        # trap the opponent if they are nearly trapped and if you can trap them on this turn
        opponent_is_trapped = self.is_almost_trapped(chess_board, adv_pos)
        if (opponent_is_trapped[0] and player_grid[opponent_is_trapped[1][0][0]][opponent_is_trapped[1][0][1]] <= max_step):
            return opponent_is_trapped[1]

        if len(barriers) > 1:
            # if there are more than 1 barriers left to build, go to the closest barrier
            if player_grid[closest[0][0]][closest[0][1]] <= max_step:
                return closest
            else:

                # If not in range, try to find the closest location to the barrier that is in range in the next turn
                for k in range(1, max_step + 1):
                    for i in range(len(player_grid)):
                        for j in range(len(player_grid)):
                            if barrier_grid[i][j] == k and player_grid[i][j] <= max_step:
                                coord = (i, j)
                                dir = self.find_dir_for_move(chess_board, player_grid, coord)
                                return (coord, dir)

                # If not in range in the next turn, find the closest location to the barrier
                location = None
                distance = 10000
                dir = 0
                for i in range(len(player_grid)):
                    for j in range(len(player_grid)):
                        if player_grid[i][j] <= max_step:
                            if location is None or barrier_grid[i][j] < distance:
                                location = (i, j)
                                distance = barrier_grid[i][j]
                                dir = self.find_dir_for_move(chess_board, player_grid, location)

                return (location, dir)

        # if there is only one barrier left to build, identify the side to build it on
        elif len(barriers) == 1:
            goal = []

            # Even board
            if m % 2 == 0:

                # Last barrier is on the top
                if (closest[0][0] == 0 and closest[0][1] == (m / 2 - 1) and closest[1] == 1) or \
                        (closest[0][0] == 0 and closest[0][1] == m / 2 and closest[1] == 3):
                    if adv_pos[1] >= m / 2:
                        goal += [((0, int(m / 2)), 1)]
                        goal += [((0, int(m / 2)), 2)]

                    else:
                        goal += [((0, int(m / 2 - 1)), 2)]
                        goal += [((0, int(m / 2 - 1)), 3)]

                # Last barrier is on the bottom
                elif (closest[0][0] == m - 1 and closest[0][1] == (m / 2 - 1) and closest[1] == 1) or \
                        (closest[0][0] == m - 1 and closest[0][1] == m / 2 and closest[1] == 3):
                    if adv_pos[1] >= m / 2:
                        goal += [((m - 1, int(m / 2)), 0)]
                        goal += [((m - 1, int(m / 2)), 1)]

                    else:
                        goal += [((m - 1, int(m / 2 - 1)), 0)]
                        goal += [((m - 1, int(m / 2 - 1)), 3)]

                # Last barrier is in the middle
                else:
                    if closest[0][1] == (m / 2 - 1):
                        if adv_pos[1] >= m / 2:
                            goal += [((closest[0][0], closest[0][1] + 1), 0)]
                            goal += [((closest[0][0], closest[0][1] + 1), 1)]
                            goal += [((closest[0][0], closest[0][1] + 1), 2)]

                        else:
                            goal += [((closest[0][0], closest[0][1]), 0)]
                            goal += [((closest[0][0], closest[0][1]), 2)]
                            goal += [((closest[0][0], closest[0][1]), 3)]

                    else:
                        if adv_pos[1] >= m / 2:
                            goal += [((closest[0][0], closest[0][1]), 0)]
                            goal += [((closest[0][0], closest[0][1]), 1)]
                            goal += [((closest[0][0], closest[0][1]), 2)]

                        else:
                            goal += [((closest[0][0], closest[0][1] - 1), 0)]
                            goal += [((closest[0][0], closest[0][1] - 1), 2)]
                            goal += [((closest[0][0], closest[0][1] - 1), 3)]

            # Odd board
            else:
                n = m // 2

                # Last barrier on the top
                if (closest[0][0] == 0 and closest[0][1] == n and closest[1] == 1) or \
                        (closest[0][0] == 0 and closest[0][1] == n + 1 and closest[1] == 3):
                    # Check if the adv is on the right of the line
                    if (adv_pos[0] in [x for x in range(0, n + 1)] and adv_pos[1] in [x for x in range(n + 1, m)]) or \
                            (adv_pos[0] in [x for x in range(n + 1, m)] and adv_pos[1] in [x for x in range(n, m)]):
                        goal += [((0, n), 1)]

                    # The adv is on the left of the line
                    else:
                        goal += [((0, n), 2)]
                        goal += [((0, n), 3)]

                # Last barrier on the bottom
                elif (closest[0][0] == m - 1 and closest[0][1] == n - 1 and closest[1] == 1) or \
                        (closest[0][0] == m - 1 and closest[0][1] == n and closest[1] == 3):
                    # Check if the adv is on the right of the line
                    if (adv_pos[0] in [x for x in range(0, n + 1)] and adv_pos[1] in [x for x in range(n + 1, m)]) or \
                            (adv_pos[0] in [x for x in range(n + 1, m)] and adv_pos[1] in [x for x in range(n, m)]):
                        goal += [((m - 1, n - 1), 1)]

                    else:
                        goal += [((m - 1, n - 1), 0)]
                        goal += [((m - 1, n - 1), 3)]

                # Last barrier in the middle
                else:
                    # Check if the position is on the left of the line
                    if (my_pos[0] in [x for x in range(0, n + 1)] and my_pos[1] in [x for x in range(0, n + 1)]) or \
                            (my_pos[0] in [x for x in range(n + 1, m)] and my_pos[1] in [x for x in range(0, n)]):
                        # Check if the position of the adv is on the left of the line
                        if (adv_pos[0] in [x for x in range(0, n + 1)] and
                            adv_pos[1] in [x for x in range(0, n + 1)]) or \
                                (adv_pos[0] in [x for x in range(n + 1, m)]
                                 and adv_pos[1] in [x for x in range(0, n)]):

                            # Check if the barrier is on the connecting part
                            if closest[0][0] == n and closest[0][1] == n and closest[1] == 2:
                                goal += [((closest[0][0], closest[0][1]), 0)]
                                goal += [((closest[0][0], closest[0][1]), 3)]

                            # Check if the barrier is on the big or the small wall
                            else:
                                goal += [((closest[0][0], closest[0][1]), 0)]
                                goal += [((closest[0][0], closest[0][1]), 2)]
                                goal += [((closest[0][0], closest[0][1]), 3)]

                        # The position of the adv is on the right of the line
                        else:
                            # Check if the barrier is on the connecting part
                            if closest[0][0] == n and closest[0][1] == n and closest[1] == 2:
                                goal += [((closest[0][0], closest[0][1]), 2)]

                            # Check if the barrier is on the big or the small wall
                            else:
                                goal += [((closest[0][0], closest[0][1]), 1)]

                    # The position is on the right of the line
                    else:
                        # Check if the position of the adv is on the left of the line
                        if (adv_pos[0] in [x for x in range(0, n + 1)] and
                            adv_pos[1] in [x for x in range(0, n + 1)]) or \
                                (adv_pos[0] in [x for x in range(n + 1, m)]
                                 and adv_pos[1] in [x for x in range(0, n)]):
                            # Check if the barrier is on the connecting part
                            if closest[0][0] == n + 1 and closest[0][1] == n and closest[1] == 0:
                                goal += [((closest[0][0] - 1, closest[0][1]), 0)]
                                goal += [((closest[0][0] - 1, closest[0][1]), 3)]

                            # The barrier is on the big or the small wall
                            else:
                                goal += [((closest[0][0], closest[0][1] - 1), 0)]
                                goal += [((closest[0][0], closest[0][1] - 1), 2)]
                                goal += [((closest[0][0], closest[0][1] - 1), 3)]

                        # The position of the adv is on the right of the line
                        else:
                            # Check if the barrier is on the connecting part
                            if closest[0][0] == n + 1 and closest[0][1] == n and closest[1] == 0:
                                goal += [((closest[0][0] - 1, closest[0][1]), 2)]

                            # The barrier is on the big or the small wall
                            else:
                                goal += [((closest[0][0], closest[0][1] - 1), 1)]

            goal_pos = goal[0][0]

            # if the goal location is reached or reachable during this turn, identify and build the remaining walls
            if ((my_pos[0] == goal_pos[0] and my_pos[1] == goal_pos[1]) or
                    (player_grid[goal_pos[0]][goal_pos[1]] <= max_step)):

                if len(goal) == 1:
                    return goal[0]

                elif len(goal) == 2:
                    if chess_board[goal_pos[0]][goal_pos[1]][goal[0][1]]:
                        return goal[1]
                    else:
                        return goal[0]

                elif len(goal) == 3:
                    if chess_board[goal_pos[0]][goal_pos[1]][goal[0][1]]:
                        if chess_board[goal_pos[0]][goal_pos[1]][goal[1][1]]:
                            return goal[2]

                        else:
                            return goal[1]

                    else:
                        return goal[0]

            else:
                # if not, move to it using the same logic as above
                goal_grid = self.board_parser(chess_board, goal_pos, adv_pos)
                for k in range(1, max_step + 1):
                    for i in range(len(player_grid)):
                        for j in range(len(player_grid)):
                            if goal_grid[i][j] == k and player_grid[i][j] <= max_step:
                                coord = (i, j)
                                dir = self.find_dir_for_move(chess_board, player_grid, coord)
                                return (coord, dir)

                # If not in range in the next turn, find the closest location to the barrier
                location = None
                distance = 10000
                dir = 0
                for i in range(len(player_grid)):
                    for j in range(len(player_grid)):
                        if player_grid[i][j] <= max_step:
                            if location is None or goal_grid[i][j] < distance:
                                location = (i, j)
                                distance = goal_grid[i][j]
                                dir = self.find_dir_for_move(chess_board, player_grid, location)

                return (location, dir)

        # If the walls are built, do random moves
        else:
            # Moves (Up, Right, Down, Left)
            ori_pos = deepcopy(my_pos)
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            steps = np.random.randint(0, max_step + 1)

            # Random Walk
            for _ in range(steps):
                r, c = my_pos
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

                # Special Case enclosed by Adversary
                k = 0
                while chess_board[r, c, dir] or my_pos == adv_pos:
                    k += 1
                    if k > 300:
                        break
                    dir = np.random.randint(0, 4)
                    m_r, m_c = moves[dir]
                    my_pos = (r + m_r, c + m_c)

                if k > 300:
                    my_pos = ori_pos
                    break

            # Put Barrier
            dir = np.random.randint(0, 4)
            r, c = my_pos
            while chess_board[r, c, dir]:
                dir = np.random.randint(0, 4)

            return my_pos, dir

    @staticmethod
    # see if a square on the board is almost trapped
    def is_almost_trapped(chess_board, pos):
        count = 0
        open = ((-1, -1), -1)
        # iterates through the directions of the position, if there is no wall it adds to the count and saves the wall position
        for i in range(4):
            if (chess_board[pos[0]][pos[1]][i] == False):
                count += 1
                if (i == 0):
                    open = ((pos[0] - 1, pos[1]), 2)
                elif (i == 1):
                    open = ((pos[0], pos[1] + 1), 3)
                elif (i == 2):
                    open = ((pos[0] + 1, pos[1]), 0)
                elif (i == 3):
                    open = ((pos[0], pos[1] - 1), 1)
        # if there is only opening left, then the position is almost trapped
        if (count == 1):
            return (True, open)
        return (False, open)

    @staticmethod
    # find the direction that the agent came from
    def find_dir_for_move(chess_board, player_grid, coord):
        if not chess_board[coord[0], coord[1], 0]:
            if player_grid[coord[0] - 1][coord[1]] == player_grid[coord[0]][coord[1]] - 1:
                return 0
        if not chess_board[coord[0], coord[1], 1]:
            if player_grid[coord[0]][coord[1] + 1] == player_grid[coord[0]][coord[1]] - 1:
                return 1
        if not chess_board[coord[0], coord[1], 2]:
            if player_grid[coord[0] + 1][coord[1]] == player_grid[coord[0]][coord[1]] - 1:
                return 2
        if not chess_board[coord[0], coord[1], 3]:
            if player_grid[coord[0]][coord[1] - 1] == player_grid[coord[0]][coord[1]] - 1:
                return 3

        if not chess_board[coord[0], coord[1], 0]:
            return 0

        if not chess_board[coord[0], coord[1], 1]:
            return 1

        if not chess_board[coord[0], coord[1], 2]:
            return 2

        if not chess_board[coord[0], coord[1], 3]:
            return 3

    @staticmethod
    def find_closest_barrier(parsed, barriers):
        min = -1
        location = ((-1, -1), -1)
        for i in barriers:
            if (min == -1 or min > parsed[i[0]][i[1]]):
                min = parsed[i[0]][i[1]]
                location = ((i[0], i[1]), i[2])

        return location

    @staticmethod
    # Find all the positions the agent can go to.
    # Return a MxM array with value True if the square can be reached by the agent, False otherwise.
    def board_parser(chess_board, my_pos, adv_pov):
        m = len(chess_board)

        k = 1000

        my_x = my_pos[0]
        my_y = my_pos[1]
        adv_x = adv_pov[0]
        adv_y = adv_pov[1]
        reachable = np.full((m, m), math.inf)
        visited = [(my_x, my_y)]

        remove_move = {0: 2, 1: 3, 2: 0, 3: 1}
        last_move = -1
        queue = [(my_x, my_y, k)]
        reachable[my_x][my_y] = 0

        while len(queue) != 0:
            moves = [0, 1, 2, 3]
            current_pos = queue.pop(0)
            cur_x = current_pos[0]
            cur_y = current_pos[1]
            cur_k = current_pos[2]

            # If there are no remaining moves at a certain position, go to the next item in the queue
            if cur_k == 0:
                continue

            # Remove the opposite direction of the last move from the moves list to prevent the agent from going back on
            # its tracks
            if last_move != -1:
                moves = moves.remove(remove_move[last_move])

            for move in moves:
                # Check if a move up is possible
                if move == 0:
                    # Check is there is no barrier and the adversary is not on the target square
                    if not chess_board[cur_x][cur_y][move] and not (cur_x - 1 == adv_x and cur_y == adv_y) \
                            and (cur_x - 1, cur_y) not in visited:
                        if k - cur_k + 1 < reachable[cur_x - 1][cur_y]:
                            reachable[cur_x - 1][cur_y] = k - cur_k + 1

                        visited += [(cur_x - 1, cur_y)]
                        queue.append((cur_x - 1, cur_y, cur_k - 1))

                # Check if a move right is possible
                elif move == 1:
                    # Check is there is no barrier and the adversary is not on the target square
                    if not chess_board[cur_x][cur_y][move] and not (cur_x == adv_x and cur_y + 1 == adv_y) \
                            and (cur_x, cur_y + 1) not in visited:
                        if k - cur_k + 1 < reachable[cur_x][cur_y + 1]:
                            reachable[cur_x][cur_y + 1] = k - cur_k + 1

                        visited += [(cur_x, cur_y + 1)]
                        queue.append((cur_x, cur_y + 1, cur_k - 1))

                # Check if a move down is possible
                elif move == 2:
                    # Check is there is no barrier and the adversary is not on the target square
                    if not chess_board[cur_x][cur_y][move] and not (cur_x + 1 == adv_x and cur_y == adv_y) \
                            and (cur_x + 1, cur_y) not in visited:
                        if k - cur_k + 1 < reachable[cur_x + 1][cur_y]:
                            reachable[cur_x + 1][cur_y] = k - cur_k + 1

                        visited += [(cur_x + 1, cur_y)]
                        queue.append((cur_x + 1, cur_y, cur_k - 1))

                # Check if a move left is possible
                elif move == 3:
                    # Check is there is no barrier and the adversary is not on the target square
                    if not chess_board[cur_x][cur_y][move] and not (cur_x == adv_x and cur_y - 1 == adv_y) \
                            and (cur_x, cur_y - 1) not in visited:
                        if k - cur_k + 1 < reachable[cur_x][cur_y - 1]:
                            reachable[cur_x][cur_y - 1] = k - cur_k + 1

                        visited += [(cur_x, cur_y - 1)]
                        queue.append((cur_x, cur_y - 1, cur_k - 1))

        return reachable

    @staticmethod
    def find_barriers_to_build(chess_board, my_pos):
        if (len(chess_board) % 2 == 0):
            walls_left = []
            # Separate cases based off of position of the agent
            if my_pos[1] < len(chess_board) / 2:
                mid = int(len(chess_board) / 2 - 1)
                for i in range(len(chess_board)):
                    # If a wall does not exist on a square along the middle of the board, add it to the list
                    if (not chess_board[i, mid, 1]):
                        walls_left.append((i, mid, 1))
            else:
                mid = int(len(chess_board) / 2)
                for i in range(len(chess_board)):
                    if (not chess_board[i, mid, 3]):
                        walls_left.append((i, mid, 3))
            return walls_left
        else:
            barriers = []
            m = len(chess_board)
            n = m // 2
            x = my_pos[0]
            y = my_pos[1]

            # Check if the position is left of the line
            if (x in [x for x in range(0, n + 1)] and y in [x for x in range(0, n + 1)]) or \
                    (x in [x for x in range(n + 1, m)] and y in [x for x in range(0, n)]):

                # Add the "big" wall
                for i in range(0, n + 1):
                    if not chess_board[i][n][1]:
                        barriers.append((i, n, 1))

                # Add the connecting part
                if not chess_board[n][n][2]:
                    barriers.append((n, n, 2))

                # Add the "small" wall
                for j in range(n + 1, m):
                    if not chess_board[j][n - 1][1]:
                        barriers.append((j, n - 1, 1))

            # Check if the position is right of the line
            else:
                # Add the "big" wall
                for i in range(0, n + 1):
                    if not chess_board[i][n + 1][3]:
                        barriers.append((i, n + 1, 3))

                # Add the connecting part
                if not chess_board[n + 1][n][0]:
                    barriers.append((n + 1, n, 0))

                # Add the "small" wall
                for j in range(n + 1, m):
                    if not chess_board[j][n][3]:
                        barriers.append((j, n, 3))

            return barriers


if __name__ == "__main__":
    agent = StudentAgent()
    chess_board = np.zeros((11, 11, 4), dtype=bool)
    chess_board[0, :, 0] = True
    chess_board[:, 0, 3] = True
    chess_board[-1, :, 2] = True
    chess_board[:, -1, 1] = True
    x = agent.board_parser(chess_board, (1, 1), (0, 0), True)
    y = agent.find_barriers_to_build(chess_board, (1, 1))
    print(x)
    print(y)
    print(agent.find_closest_barrier(x, y))
