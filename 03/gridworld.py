import numpy as np
import sys
from contextlib import closing
from io import StringIO

# Constants for cardinal directions with more descriptive names
DIRECTION_UP = 0
DIRECTION_RIGHT = 1
DIRECTION_DOWN = 2
DIRECTION_LEFT = 3

class GridworldEnv():
    """
    A 4x4 Grid World environment from Sutton's Reinforcement 
    Learning book chapter 4. Terminal states are top left and
    the bottom right corner.

    Actions are (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave agent in current state.
    Reward of -1 at each step until agent reaches a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        # Define grid dimensions
        self.grid_shape = (4, 4)
        
        # Calculate total number of states and actions
        self.number_of_states = np.prod(self.grid_shape)
        self.number_of_actions = 4

        # Dictionary to store state transition dynamics
        transitions = {}
        for state in range(self.number_of_states):
            # Convert 1D state index to 2D grid coordinates
            grid_position = np.unravel_index(state, self.grid_shape)
            # Initialize transitions for each action
            transitions[state] = {action: [] for action in range(self.number_of_actions)}
            # Calculate transition probabilities for each action
            transitions[state][DIRECTION_UP] = self.transition_prob(grid_position, [-1, 0])
            transitions[state][DIRECTION_RIGHT] = self.transition_prob(grid_position, [0, 1])
            transitions[state][DIRECTION_DOWN] = self.transition_prob(grid_position, [1, 0])
            transitions[state][DIRECTION_LEFT] = self.transition_prob(grid_position, [0, -1])

        # Set uniform initial state distribution
        initial_state_distribution = np.ones(self.number_of_states) / self.number_of_states

        # Store transition dynamics for model-based learning
        self.transitions = transitions

    def limit_coordinates(self, coordinates):
        """
        Prevent the agent from falling out of the grid world by clamping coordinates
        :param coordinates: a tuple(x,y) position on the grid
        :return: new coordinates ensuring that they are within the grid world
        """
        # Clamp row coordinate
        coordinates[0] = min(coordinates[0], self.grid_shape[0] - 1)
        coordinates[0] = max(coordinates[0], 0)
        # Clamp column coordinate
        coordinates[1] = min(coordinates[1], self.grid_shape[1] - 1)
        coordinates[1] = max(coordinates[1], 0)
        return coordinates

    def transition_prob(self, current_position, movement_delta):
        """
        Model Transitions. Probability is always 1.0 (deterministic environment)
        :param current_position: Current position on the grid as (row, col)
        :param movement_delta: Change in position for transition [δrow, δcol]
        :return: [(1.0, new_state, reward, done)] - List with single transition tuple
        """
        # Convert current position to state index
        current_state = np.ravel_multi_index(tuple(current_position), self.grid_shape)
        
        # Handle terminal states
        if current_state == 0 or current_state == self.number_of_states - 1:
            return [(1.0, current_state, 0, True)]

        # Calculate new position
        new_position = np.array(current_position) + np.array(movement_delta)
        new_position = self.limit_coordinates(new_position).astype(int)
        next_state = np.ravel_multi_index(tuple(new_position), self.grid_shape)

        # Check if new state is terminal
        is_terminal = next_state == 0 or next_state == self.number_of_states - 1
        return [(1.0, next_state, -1, is_terminal)]

    def render(self, render_mode='human'):
        """
        Render the grid world environment
        :param render_mode: 'human' for stdout or 'ansi' for string buffer
        :return: None for 'human' mode, string for 'ansi' mode
        """
        output_file = StringIO() if render_mode == 'ansi' else sys.stdout

        for state in range(self.number_of_states):
            grid_position = np.unravel_index(state, self.grid_shape)
            if self.current_state == state:
                cell_display = " x "  # Agent position
            elif state == 0 or state == self.number_of_states - 1:
                cell_display = " T "  # Terminal states
            else:
                cell_display = " o "  # Empty cells

            # Format grid layout
            if grid_position[1] == 0:
                cell_display = cell_display.lstrip()
            if grid_position[1] == self.grid_shape[1] - 1:
                cell_display = cell_display.rstrip()
                cell_display += '\n'

            output_file.write(cell_display)
        output_file.write('\n')

        if render_mode != 'human':
            with closing(output_file):
                return output_file.getvalue()