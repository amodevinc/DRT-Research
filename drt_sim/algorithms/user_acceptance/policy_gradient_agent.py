import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class PolicyGradientAgent:
    """
    Policy gradient agent that learns to predict user choices.
    
    This agent uses a softmax-based choice model to select alternatives based
    on learned weights. It updates its policy using the REINFORCE algorithm
    with discounted returns.
    
    Attributes:
        alpha (float): Learning rate for gradient ascent.
        beta (float): Sensitivity parameter for the logit (softmax) function.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration rate for ε-greedy action selection.
        weights (ndarray): Learned weights of shape (feature_dim,).
        weight_history (list): List of weight snapshots during training.
        reward_history (list): List of episode rewards during training.
        accuracy_history (list): List of episode accuracy values during training.
    """
    
    def __init__(self, alpha, beta, feature_dim, epsilon, gamma=0.99):
        """
        Initialize the agent with hyperparameters.
        
        Args:
            alpha (float): Learning rate.
            beta (float): Softmax temperature parameter.
            feature_dim (int): Dimensionality of the feature vectors.
            epsilon (float): Exploration probability for ε-greedy policy.
            gamma (float, optional): Discount factor. Defaults to 0.99.
        """
        self.alpha = alpha  # Learning rate
        self.beta = beta    # Logit model sensitivity parameter
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.weights = np.zeros((feature_dim,))  # Initialize weights to zeros
        
        # For visualization purposes
        self.weight_history = [self.weights.copy()]
        self.reward_history = []
        self.accuracy_history = []
        self.feature_names = None  # Will be set if feature names are available

    def set_feature_names(self, feature_names):
        """
        Set the names of features for better visualization.
        
        Args:
            feature_names (list): List of feature names.
        """
        self.feature_names = feature_names

    def softmax(self, action_values):
        """
        Calculate action probabilities using the softmax function.
        
        This implementation includes numerical stability improvements by
        subtracting the maximum value before exponentiation.
        
        Args:
            action_values (ndarray): Values for each action.
            
        Returns:
            ndarray: Probability distribution over actions.
        """
        max_value = np.max(action_values)  # For numerical stability
        exp_values = np.exp(self.beta * (action_values - max_value))  # Prevent overflow
        return exp_values / np.sum(exp_values)

    def select_action(self, situation, features):
        """
        Select an action based on the current policy.
        
        Uses ε-greedy strategy: with probability epsilon selects a random action,
        otherwise selects the action with highest probability according to the policy.
        
        Args:
            situation (int): Current situation index.
            features (ndarray): Feature array of shape [num_situations, num_alternatives, feature_dim].
            
        Returns:
            tuple: (action, action_probabilities) where:
                - action (int): The selected alternative.
                - action_probabilities (ndarray): Probability distribution over alternatives.
        """
        if np.random.rand() < self.epsilon:
            # Random exploration
            num_alternatives = len(features[situation])
            return np.random.choice(np.arange(num_alternatives)), [1/num_alternatives] * num_alternatives
        else:
            # Exploitation based on learned policy
            action_values = np.dot(features[situation], self.weights)
            action_probabilities = self.softmax(action_values)
            action = np.argmax(action_probabilities)
            return action, action_probabilities
    
    def policy_evaluation(self, history):
        """
        Evaluate the policy by computing returns for each time step.
        
        Works backwards through the history to compute discounted returns.
        
        Args:
            history (list): List of (action, state, reward, action_probabilities) tuples.
            
        Returns:
            ndarray: Array of returns (G) for each time step.
        """
        G = 0
        returns = []
        for _, _, reward, _ in reversed(history):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        return returns

    def policy_improvement(self, history, returns, features):
        """
        Improve the policy by gradient ascent on the weights.
        
        For each time step, computes the gradient of log-prob(action) * return
        and updates the weights accordingly.
        
        Args:
            history (list): List of (action, state, reward, action_probabilities) tuples.
            returns (ndarray): Array of returns (G) for each time step.
            features (ndarray): Feature array of shape [num_situations, num_alternatives, feature_dim].
        """
        for (action, state, _, action_probabilities), G in zip(history, returns):
            gradient = np.zeros_like(self.weights)
            for a in range(len(action_probabilities)):
                if a == action:
                    # Gradient for the chosen action
                    gradient += (1 - action_probabilities[a]) * features[state][a]
                else:
                    # Gradient for unchosen actions
                    gradient -= action_probabilities[a] * features[state][a]
            # Update weights in the direction of the gradient
            self.weights += self.alpha * G * gradient

    def update_policy(self, history, features):
        """
        Update the policy based on the episode history.
        
        This is a convenience method that calls policy_evaluation to get returns
        and then calls policy_improvement to update the weights.
        
        Args:
            history (list): List of (action, state, reward, action_probabilities) tuples.
            features (ndarray): Feature array of shape [num_situations, num_alternatives, feature_dim].
        """
        returns = self.policy_evaluation(history)
        self.policy_improvement(history, returns, features)
        
        # Store weight snapshot for visualization
        self.weight_history.append(self.weights.copy())
        
        # Calculate and store total reward
        total_reward = sum(item[2] for item in history)
        self.reward_history.append(total_reward)
        
        # Calculate episode accuracy
        correct_predictions = sum(1 for item in history if item[2] == 1)
        accuracy = correct_predictions / len(history)
        self.accuracy_history.append(accuracy)

    def print_weights(self):
        """Print the learned weights."""
        print("Learned weights:")
        if self.feature_names:
            for name, weight in zip(self.feature_names, self.weights):
                print(f"  {name}: {weight:.6f}")
        else:
            print(self.weights)

    def visualize_weights(self, save_path=None):
        """
        Visualize the current weights as a bar chart.
        
        Args:
            save_path (str, optional): Path to save the figure. If None, the figure is displayed.
        """
        plt.figure(figsize=(12, 6))
        
        if self.feature_names:
            feature_indices = np.arange(len(self.weights))
            plt.bar(feature_indices, self.weights)
            plt.xticks(feature_indices, self.feature_names, rotation=45, ha='right')
        else:
            feature_indices = np.arange(len(self.weights))
            plt.bar(feature_indices, self.weights)
            plt.xticks(feature_indices, [f'Feature {i}' for i in feature_indices])
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Feature Weights')
        plt.ylabel('Weight Value')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def visualize_learning_progress(self, save_path=None):
        """
        Visualize the learning progress during training.
        
        Creates a figure with two subplots: one for total reward per episode
        and another for prediction accuracy per episode.
        
        Args:
            save_path (str, optional): Path to save the figure. If None, the figure is displayed.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        episodes = range(1, len(self.reward_history) + 1)
        
        # Plot total reward
        ax1.plot(episodes, self.reward_history, marker='o', linestyle='-')
        ax1.set_title('Total Reward per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(episodes, self.accuracy_history, marker='o', linestyle='-', color='green')
        ax2.set_title('Prediction Accuracy per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def create_weight_evolution_animation(self, save_path=None):
        """
        Create an animation showing how weights evolve during training.
        
        Args:
            save_path (str, optional): Path to save the animation as a gif. 
                                     If None, the animation is displayed.
        
        Returns:
            matplotlib.animation.FuncAnimation: Animation object.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        feature_dim = len(self.weights)
        if self.feature_names:
            feature_labels = self.feature_names
        else:
            feature_labels = [f'Feature {i}' for i in range(feature_dim)]
            
        # Set up the initial bar plot
        x = np.arange(feature_dim)
        bars = ax.bar(x, self.weight_history[0])
        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels, rotation=45, ha='right')
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax.set_title('Feature Weights Evolution')
        ax.set_ylabel('Weight Value')
        
        y_min = min(np.min(w) for w in self.weight_history)
        y_max = max(np.max(w) for w in self.weight_history)
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
        
        # Text for episode number
        episode_text = ax.text(0.02, 0.95, 'Episode: 0', transform=ax.transAxes)
        
        def update(frame):
            # Update the heights of the bars
            for i, bar in enumerate(bars):
                bar.set_height(self.weight_history[frame][i])
            # Update the episode text
            episode_text.set_text(f'Episode: {frame}')
            
            # Convert bars to a list before concatenating with episode_text
            return list(bars) + [episode_text]
        
        anim = FuncAnimation(fig, update, frames=len(self.weight_history), 
                              blit=True, interval=200, repeat=True)
        
        plt.tight_layout()
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
            plt.close()
            return None
        else:
            plt.close()
            return HTML(anim.to_jshtml())

    def visualize_choice_probabilities(self, features, user_history, save_path=None):
        """
        Visualize the choice probabilities for each situation.
        
        For each situation, shows the probability the model assigns to each alternative
        and highlights the user's actual choice.
        
        Args:
            features (ndarray): 3D array of features.
            user_history (list): List of the user's actual choices.
            save_path (str, optional): Path to save the figure. If None, the figure is displayed.
        """
        num_situations = min(len(user_history), 10)  # Limit to first 10 situations for clarity
        num_alternatives = features[0].shape[0]
        
        fig, axes = plt.subplots(num_situations, 1, figsize=(12, num_situations * 2))
        if num_situations == 1:
            axes = [axes]
        
        for i in range(num_situations):
            # Get probabilities for this situation
            action_values = np.dot(features[i], self.weights)
            probs = self.softmax(action_values)
            
            # Create bar colors (highlight actual choice)
            colors = ['lightblue'] * num_alternatives
            colors[user_history[i]] = 'orange'
            
            # Plot
            ax = axes[i]
            bars = ax.bar(range(num_alternatives), probs, color=colors)
            ax.set_title(f'Situation {i}')
            ax.set_ylim(0, 1)
            ax.set_xticks(range(num_alternatives))
            ax.set_xticklabels([f'Alt {j}' for j in range(num_alternatives)])
            ax.set_ylabel('Probability')
            
            # Add text labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{probs[j]:.2f}', ha='center', va='bottom')
                
                # Mark actual choice
                if j == user_history[i]:
                    ax.text(bar.get_x() + bar.get_width()/2., height/2,
                            'Actual\nChoice', ha='center', va='center', 
                            color='black', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()