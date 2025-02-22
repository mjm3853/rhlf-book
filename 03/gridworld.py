import numpy as np
import sys
from contextlib import closing
from io import StringIO

# Constants for cardinal directions
# Mapped to indices for array operations
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv():
    """
    A 4x4 Grid World environment from Sutton's Reinforcement 
    Learning book chapter 4. Terminal states are top left and
    the bottom right corner.

    Actions are (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave agent in current state.
    Reward of -1 at each step until agent reaches a terminal state.
    """

    # Supported rendering modes for the environment
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        # Define grid dimensions (4x4)
        self.shape = (4, 4)
        
        # Calculate total number of states (16 for 4x4 grid)
        self.nS = np.prod(self.shape)
        # Number of possible actions in each state
        self.nA = 4

        # Initialize current state
        self.s = None  # Current state of the agent

        # Dictionary to store state transition dynamics
        P = {}
        for s in range(self.nS):
            # Convert 1D state index to 2D grid coordinates
            position = np.unravel_index(s, self.shape)
            # Initialize transitions for each action
            P[s] = {a: [] for a in range(self.nA)}
            # Calculate transition probabilities for each action
            # Format: (probability, next_state, reward, done)
            P[s][UP] = self.transition_prob(position, [-1, 0])      # Move up
            P[s][RIGHT] = self.transition_prob(position, [0, 1])    # Move right
            P[s][DOWN] = self.transition_prob(position, [1, 0])     # Move down
            P[s][LEFT] = self.transition_prob(position, [0, -1])    # Move left

        # Set uniform initial state distribution
        isd = np.ones(self.nS) / self.nS

        # Store transition dynamics for model-based learning
        self.P = P

    def limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world by clamping coordinates
        :param coord: a tuple(x,y) position on the grid
        :return: new coordinates ensuring that they are within the grid world
        """
        # Clamp row coordinate between 0 and max row index
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        # Clamp column coordinate between 0 and max column index
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def transition_prob(self, current, delta):
        """
        Model Transitions. Probability is always 1.0 (deterministic environment)
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition [δrow, δcol]
        :return: [(1.0, new_state, reward, done)] - List with single transition tuple
        """
        # Convert current 2D position to 1D state index
        current_state = np.ravel_multi_index(tuple(current), self.shape)
        
        # Handle terminal states (top-left and bottom-right corners)
        if current_state == 0 or current_state == self.nS - 1:
            return [(1.0, current_state, 0, True)]

        # Calculate new position by adding movement delta
        new_position = np.array(current) + np.array(delta)
        # Ensure new position is within grid boundaries
        new_position = self.limit_coordinates(new_position).astype(int)
        # Convert new 2D position to 1D state index
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        # Check if new state is terminal
        is_done = new_state == 0 or new_state == self.nS - 1
        # Return transition tuple with -1 reward for non-terminal moves
        return [(1.0, new_state, -1, is_done)]

    def render(self, mode='human'):
        """
        Render the grid world environment
        :param mode: 'human' for stdout or 'ansi' for string buffer
        :return: None for 'human' mode, string for 'ansi' mode
        """
        # Select output destination based on mode
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        # Iterate through all states to create grid visualization
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "  # Mark current agent position
            elif s == 0 or s == self.nS - 1:
                output = " T "  # Mark terminal states
            else:
                output = " o "  # Mark empty cells

            # Format grid layout
            if position[1] == 0:  # Left edge
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:  # Right edge
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')

        # Return string representation for ANSI mode
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def reset(self):
        """
        Reset the environment to a random non-terminal state
        :return: Initial state
        """
        # Initialize to random non-terminal state
        while True:
            self.s = np.random.randint(0, self.nS)
            # Ensure we don't start in terminal states
            if self.s != 0 and self.s != self.nS - 1:
                break
        return self.s