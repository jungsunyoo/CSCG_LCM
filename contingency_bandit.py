import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class DriftingBandit:
    def __init__(self, n_arms: int, drift_rate: float = 0.05, noise: float = 0.2):
        """Initialize a drifting multi-arm bandit environment.
        
        Args:
            n_arms: Number of arms in the bandit
            drift_rate: Rate at which probabilities drift (for drifting condition)
            noise: Standard deviation of drift noise
        """
        self.n_arms = n_arms
        self.drift_rate = drift_rate
        self.noise = noise
        self.probs = np.random.rand(n_arms)  # Initial probabilities
        self.t = 0
        
    def step(self, action: int) -> Tuple[float, bool]:
        """Take a step in the environment.
        
        Args:
            action: Index of arm to pull
            
        Returns:
            Tuple of (reward, done)
        """
        # Drift probabilities with more extreme changes
        drift = np.random.normal(0, self.noise, self.n_arms)
        # Add occasional jumps
        if np.random.random() < 0.05:  # 5% chance of jump
            drift *= 3.0
        self.probs += drift * self.drift_rate
        self.probs = np.clip(self.probs, 0, 1)  # Keep probabilities between 0 and 1
        
        # Get reward
        reward = np.random.binomial(1, self.probs[action])
        self.t += 1
        return reward, False
    
    def abrupt_change(self, arm_idx: int, new_prob: float):
        """Introduce an abrupt change in probability for a specific arm."""
        self.probs[arm_idx] = new_prob

class ContingencyModel:
    def __init__(self, 
                 n_arms: int,
                 window_size: int = 50,
                 split_threshold: float = 0.2,
                 deactivate_threshold: float = 0.1,
                 learning_rate: float = 0.1):
        """Initialize the contingency-dependent state-space model.
        
        Args:
            n_arms: Number of arms in the bandit
            window_size: Size of sliding window for contingency calculation
            split_threshold: Threshold for splitting new states
            deactivate_threshold: Threshold for deactivating states
            learning_rate: Learning rate for value updates
        """
        self.n_arms = n_arms
        self.window_size = window_size
        self.split_threshold = split_threshold
        self.deactivate_threshold = deactivate_threshold
        self.learning_rate = learning_rate
        
        # Initialize state tracking
        self.states = [0]  # List of active states
        self.state_values = {0: np.zeros(n_arms)}  # Value estimates for each state
        self.state_counts = {0: {'CS': 0, 'US': 0}}  # Counts for contingency calculation
        
        # History tracking
        self.history = {
            'CS': [],  # CS occurrences
            'US': [],  # US occurrences
            'states': [],  # Active states at each step
            'values': []  # Value estimates at each step
        }
        
    def calculate_prospective_contingency(self, state: int) -> float:
        """Calculate P(US|CS) for a given state."""
        counts = self.state_counts[state]
        if counts['CS'] == 0:
            return 0.5  # Neutral prior
        return counts['US'] / counts['CS']
    
    def calculate_retrospective_contingency(self, state: int) -> float:
        """Calculate P(CS|US) for a given state."""
        counts = self.state_counts[state]
        if counts['US'] == 0:
            return 0.5  # Neutral prior
        return counts['CS'] / counts['US']
    
    def update(self, action: int, reward: float):
        """Update the model based on action and reward.
        
        Args:
            action: Index of arm pulled
            reward: Reward received
        """
        # Update history
        self.history['CS'].append(action)
        self.history['US'].append(reward)
        
        # Update counts for current state
        current_state = self.states[-1]
        self.state_counts[current_state]['CS'] += 1
        if reward == 1:
            self.state_counts[current_state]['US'] += 1
            
        # Check for state splitting
        prospective = self.calculate_prospective_contingency(current_state)
        if len(self.history['CS']) >= self.window_size:
            window_prospective = self.calculate_prospective_contingency(current_state)
            if abs(window_prospective - prospective) > self.split_threshold:
                # Split new state
                new_state = len(self.states)
                self.states.append(new_state)
                self.state_values[new_state] = self.state_values[current_state].copy()
                self.state_counts[new_state] = {'CS': 0, 'US': 0}
                current_state = new_state
        
        # Update value estimates
        self.state_values[current_state][action] += self.learning_rate * (
            reward - self.state_values[current_state][action]
        )
        
        # Check for state deactivation
        retrospective = self.calculate_retrospective_contingency(current_state)
        if retrospective < self.deactivate_threshold and len(self.states) > 1:
            self.states.remove(current_state)
            del self.state_values[current_state]
            del self.state_counts[current_state]
            
        # Update history
        self.history['states'].append(self.states.copy())
        self.history['values'].append({
            state: self.state_values[state].copy() 
            for state in self.states
        })
    
    def select_action(self) -> int:
        """Select action using epsilon-greedy policy."""
        current_state = self.states[-1]
        values = self.state_values[current_state]
        
        # Epsilon-greedy with epsilon = 0.1
        if np.random.random() < 0.1:
            return np.random.randint(self.n_arms)
        return np.argmax(values)

def run_experiment(n_arms: int = 2,
                  n_steps: int = 1000,
                  condition: str = 'drifting',
                  drift_rate: float = 0.05,
                  noise: float = 0.2):
    """Run experiment with either drifting or abrupt change condition."""
    
    # Initialize environment
    if condition == 'drifting':
        env = DriftingBandit(n_arms, drift_rate, noise)
    else:  # abrupt change
        env = DriftingBandit(n_arms, drift_rate=0, noise=0)
        # Schedule more frequent and extreme changes
        # Changes every ~100 steps with more extreme probabilities
        change_times = list(range(100, n_steps, 100))
        change_probs = []
        for _ in range(len(change_times)):
            # Alternate between very high and very low probabilities
            if len(change_probs) % 2 == 0:
                prob = np.random.uniform(0.8, 0.95)  # Very high probability
            else:
                prob = np.random.uniform(0.05, 0.2)  # Very low probability
            change_probs.append(prob)
    
    # Initialize model with adjusted parameters for more volatile environment
    model = ContingencyModel(
        n_arms,
        window_size=30,  # Smaller window to detect changes faster
        split_threshold=0.15,  # Lower threshold to split more readily
        deactivate_threshold=0.05,  # Lower threshold to maintain states longer
        learning_rate=0.2  # Higher learning rate to adapt faster
    )
    
    # Run experiment
    rewards = []
    for t in range(n_steps):
        # Handle abrupt changes
        if condition == 'abrupt' and t in change_times:
            idx = change_times.index(t)
            # Change multiple arms with opposite probabilities
            env.abrupt_change(0, change_probs[idx])
            env.abrupt_change(1, 1 - change_probs[idx])  # Opposite probability for other arm
        
        # Select and take action
        action = model.select_action()
        reward, _ = env.step(action)
        
        # Update model
        model.update(action, reward)
        rewards.append(reward)
    
    return rewards, model.history

def plot_results(rewards: List[float], history: Dict):
    """Plot results from experiment."""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards Over Time')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    
    # Plot number of active states
    plt.subplot(2, 2, 2)
    n_states = [len(states) for states in history['states']]
    plt.plot(n_states)
    plt.title('Number of Active States')
    plt.xlabel('Step')
    plt.ylabel('Number of States')
    
    # Plot value estimates for first arm
    plt.subplot(2, 2, 3)
    for state in range(max([max(states) for states in history['states']]) + 1):
        values = [v[state][0] if state in v else None for v in history['values']]
        plt.plot(values, label=f'State {state}')
    plt.title('Value Estimates for First Arm')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    
    # Plot contingency measures
    plt.subplot(2, 2, 4)
    prospective = [model.calculate_prospective_contingency(states[-1]) 
                  for states in history['states']]
    retrospective = [model.calculate_retrospective_contingency(states[-1]) 
                    for states in history['states']]
    plt.plot(prospective, label='Prospective')
    plt.plot(retrospective, label='Retrospective')
    plt.title('Contingency Measures')
    plt.xlabel('Step')
    plt.ylabel('Contingency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run experiments for both conditions
    print("Running drifting condition...")
    rewards_drift, history_drift = run_experiment(
        n_arms=2,
        n_steps=1000,
        condition='drifting',
        drift_rate=0.05,
        noise=0.2
    )
    plot_results(rewards_drift, history_drift)
    
    print("Running abrupt change condition...")
    rewards_abrupt, history_abrupt = run_experiment(
        n_arms=2,
        n_steps=1000,
        condition='abrupt'
    )
    plot_results(rewards_abrupt, history_abrupt) 