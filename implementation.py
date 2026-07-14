import numpy as np
import torch
from torch.distributions import Exponential

class MarkovProcessWithResetting:
    def __init__(self, reset_rate, jump_rate, jump_scale, reset_value):
        """
        Initialize the Markov process with resetting.

        Args:
            reset_rate (float): Rate at which the process resets.
            jump_rate (float): Rate of exponential jumps.
            jump_scale (float): Scale parameter for the exponential jump distribution.
            reset_value (float): Value to reset to when resetting occurs.
        """
        self.reset_rate = reset_rate
        self.jump_rate = jump_rate
        self.jump_scale = jump_scale
        self.reset_value = reset_value

    def simulate(self, T, dt):
        """
        Simulate the Markov process with resetting over time.

        Args:
            T (float): Total simulation time.
            dt (float): Time step for simulation.

        Returns:
            torch.Tensor: Simulated process values over time.
        """
        time_steps = int(T / dt)
        process = torch.zeros(time_steps)
        current_value = 0.0

        for t in range(1, time_steps):
            if np.random.rand() < self.reset_rate * dt:
                # Reset occurs
                current_value = self.reset_value
            else:
                # Jump occurs
                if np.random.rand() < self.jump_rate * dt:
                    jump = Exponential(self.jump_scale).sample().item()
                    current_value += jump

            process[t] = current_value

        return process

    def invariant_density(self, x):
        """
        Compute the invariant density of the process.

        Args:
            x (torch.Tensor): Points at which to evaluate the density.

        Returns:
            torch.Tensor: Invariant density values at the given points.
        """
        lambda_r = self.reset_rate
        lambda_j = self.jump_rate
        beta = 1 / self.jump_scale

        density = lambda_r * torch.exp(-beta * x) / (lambda_r + lambda_j)
        return density

if __name__ == '__main__':
    # Parameters for the Markov process
    reset_rate = 0.5
    jump_rate = 1.0
    jump_scale = 2.0
    reset_value = 0.0
    T = 10.0  # Total simulation time
    dt = 0.01  # Time step

    # Initialize the process
    process = MarkovProcessWithResetting(reset_rate, jump_rate, jump_scale, reset_value)

    # Simulate the process
    simulated_process = process.simulate(T, dt)

    # Compute the invariant density
    x = torch.linspace(0, 10, 100)
    density = process.invariant_density(x)

    # Print results
    print("Simulated Process (first 10 values):", simulated_process[:10])
    print("Invariant Density (first 10 values):", density[:10])