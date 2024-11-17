import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger


# =========================================
# Bandit Abstract Class
# =========================================
class Bandit(ABC):
    """
    Abstract base class for a bandit in the multi-armed bandit experiment.

    Attributes:
        p: Underlying reward probability.
        name: Identifier for the bandit (e.g., "Bandit-1").
        successes: Number of successful pulls.
        failures: Number of failed pulls.
        n: Total number of pulls for this bandit.
        rewards: List of rewards received from this bandit.
    """

    def __init__(self, p, name):
        self.p = p  # Reward probability
        self.name = name  # Identifier for the bandit
        self.successes = 0  # Count of successful pulls
        self.failures = 0  # Count of failed pulls
        self.n = 0  # Total number of pulls
        self.rewards = []  # Track rewards for visualization and analysis

    def __repr__(self):
        return f"Bandit(name={self.name}, p={self.p}, successes={self.successes}, failures={self.failures}, pulls={self.n})"

    @abstractmethod
    def pull(self):
        """
        Simulates pulling the bandit's arm and returns the reward.
        """
        pass

    @abstractmethod
    def update(self, reward):
        """
        Updates the bandit's parameters based on the received reward.
        """
        pass

    @abstractmethod
    def experiment(self, num_trials):
        """
        Runs the experiment by simulating multiple pulls on this bandit.
        """
        pass

    def report(self, algorithm_name):
        """
        Generates a detailed report for the bandit's performance.

        - Saves results (bandit, reward, algorithm) to a CSV file.
        - Logs cumulative reward and regret.

        Parameters:
            algorithm_name (str): The name of the algorithm (e.g., "Epsilon-Greedy", "Thompson Sampling").
        """
        # Save results to a CSV file
        records = [{"Bandit": self.name, "Reward": r, "Algorithm": algorithm_name} for r in self.rewards]
        filename = f"{algorithm_name}_{self.name}_results.csv"
        save_results_to_csv(records, filename)

        # Compute cumulative reward and regret
        cumulative_reward = np.sum(self.rewards)
        optimal_reward = max([b.p for b in Bandit_Reward]) * len(self.rewards)
        cumulative_regret = optimal_reward - cumulative_reward

        # Log results
        logger.info(f"Algorithm: {algorithm_name}, Bandit: {self.name}")
        logger.info(f"Cumulative Reward: {cumulative_reward}")
        logger.info(f"Cumulative Regret: {cumulative_regret}")

        # Plot the learning process
        self.plot_learning_process(algorithm_name)

    def plot_learning_process(self, algorithm_name):
        """
        Visualizes the learning process by plotting the cumulative average reward over time.
        """
        cumulative_average = np.cumsum(self.rewards) / (np.arange(1, len(self.rewards) + 1))
        plt.plot(cumulative_average, label=self.name)
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Average Reward")
        plt.title(f"{algorithm_name} - Learning Process")
        plt.legend()
        plt.show()


# =========================================
# Epsilon-Greedy Algorithm
# =========================================
class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy algorithm for multi-armed bandit problems.

    Attributes:
        epsilon: The probability of exploring a random bandit.
        estimated_mean: The current estimate of the bandit's reward rate.
    """

    def __init__(self, p, epsilon, name):
        super().__init__(p, name)
        self.epsilon = epsilon
        self.estimated_mean = 0  # Initial estimated mean reward

    def pull(self):
        """
        Simulates pulling the bandit's arm and returns a reward sampled from its probability.
        """
        return np.random.rand() < self.p

    def update(self, reward):
        """
        Updates the bandit's estimated mean reward based on the received reward.
        """
        self.n += 1
        self.rewards.append(reward)
        self.estimated_mean = ((self.n - 1) * self.estimated_mean + reward) / self.n

    def experiment(self, num_trials):
        """
        Runs the experiment for a specified number of trials using the Epsilon-Greedy strategy.
        """
        for t in range(1, num_trials + 1):
            if np.random.rand() < self.epsilon:
                choice = np.random.choice(len(Bandit_Reward))  # Explore a random bandit
            else:
                choice = np.argmax([bandit.estimated_mean for bandit in Bandit_Reward])  # Exploit the best-known bandit

            reward = Bandit_Reward[choice].pull()
            Bandit_Reward[choice].update(reward)

            # Adaptive epsilon decay
            self.epsilon = max(0.1, self.epsilon * 0.99)


# =========================================
# Thompson Sampling Algorithm
# =========================================
class ThompsonSampling(Bandit):
    """
    Thompson Sampling algorithm for multi-armed bandit problems.

    Attributes:
        successes: Number of successful pulls for this bandit.
        failures: Number of failed pulls for this bandit.
    """

    def __init__(self, p, name):
        super().__init__(p, name)

    def pull(self):
        """
        Simulates pulling the bandit's arm and returns a reward sampled from its probability.
        """
        return np.random.rand() < self.p

    def update(self, reward):
        """
        Updates the success or failure counts based on the received reward.
        """
        self.n += 1
        self.rewards.append(reward)
        if reward:
            self.successes += 1
        else:
            self.failures += 1

    def experiment(self, num_trials):
        """
        Runs the experiment for a specified number of trials using Thompson Sampling.
        """
        for _ in range(num_trials):
            samples = [np.random.beta(1 + b.successes, 1 + b.failures) for b in Bandit_Reward]
            choice = np.argmax(samples)
            reward = Bandit_Reward[choice].pull()
            Bandit_Reward[choice].update(reward)


# =========================================
# Utilities
# =========================================
def save_results_to_csv(records, filename):
    """
    Saves experiment results to a CSV file.

    Parameters:
        records (list of dict): A list of records, where each record is a dictionary containing bandit info, rewards, and algorithm.
        filename (str): The name of the CSV file.
    """
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    logger.info(f"Results saved successfully to {filename}")


# =========================================
# Main Experiment
# =========================================
if __name__ == "__main__":
    logger.info("Starting Multi-Armed Bandit Experiment...")

    # Initialize bandits with fixed probabilities
    Bandit_Reward = [EpsilonGreedy(p=np.random.uniform(0.1, 0.9), epsilon=0.1, name=f"Bandit-{i+1}")
                     for i in range(4)]

    num_trials = 20000

    # Run Epsilon-Greedy Experiment
    logger.info("Running Epsilon-Greedy Experiment...")
    for bandit in Bandit_Reward:
        bandit.experiment(num_trials)
        bandit.report("Epsilon-Greedy")

    # Select the best bandit for Epsilon-Greedy
    best_bandit_eg = max(Bandit_Reward, key=lambda b: b.estimated_mean)
    logger.info(f"Best Bandit for Epsilon-Greedy: {best_bandit_eg.name} with estimated mean reward: {best_bandit_eg.estimated_mean}")

    # Reinitialize bandits for Thompson Sampling
    logger.info("Reinitializing bandits for Thompson Sampling...")
    Bandit_Reward = [ThompsonSampling(p=np.random.uniform(0.1, 0.9), name=f"Bandit-{i+1}")
                     for i in range(4)]

    # Run Thompson Sampling Experiment
    logger.info("Running Thompson Sampling Experiment...")
    for bandit in Bandit_Reward:
        bandit.experiment(num_trials)
        bandit.report("Thompson Sampling")

    # Select the best bandit for Thompson Sampling
    best_bandit_ts = max(Bandit_Reward, key=lambda b: b.successes / max(1, b.n))  # Avoid division by zero
    logger.info(f"Best Bandit for Thompson Sampling: {best_bandit_ts.name} with success rate: {best_bandit_ts.successes / max(1, best_bandit_ts.n)}")

    logger.info("Experiment Completed.")
