import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import os

"""
Reinforcement Learning Model for User Choice Prediction with Visualization

This module implements a policy gradient approach to predict user choices based on historical data,
with added visualization components to track model performance and internal representations.

The main components are:
1. Environment - Simulates the choice environment and provides rewards
2. PolicyGradientAgent - Implements the learning algorithm with visualization capabilities
3. Data processing functions - Load and preprocess user choice data
4. Training and evaluation functions - Train the model and assess its performance
5. Visualization utilities - Create plots and animations to monitor learning
"""


class Environment:
    """
    Simulates the reinforcement learning environment for user choice behavior.
    
    This class tracks the current situation and provides a reward signal based on
    how well the agent's predicted choice matches the user's actual choice.
    
    Attributes:
        user_history (list): List of the user's actual choices for each situation.
        features (ndarray): 3D array of shape [num_situations, num_alternatives, feature_dim].
        x (int): Current situation index.
    """
    
    def __init__(self, user_history, features):
        """
        Initialize the environment with user history and features.
        
        Args:
            user_history (list): List of the user's actual choices for each situation.
            features (ndarray): 3D array of shape [num_situations, num_alternatives, feature_dim].
        """
        self.user_history = user_history
        self.features = features  # Features for each situation and alternative
        self.x = 0  # Current situation index
        
    def get_actual_choice(self):
        """
        Return the actual choice made by the user in the current situation.
        
        Returns:
            int: The alternative chosen by the user in the current situation.
        """
        return self.user_history[self.x]

    def step(self, action, actual_choice):
        """
        Compares the agent's action with the user's actual choice and returns reward.
        
        Args:
            action (int): The agent's chosen alternative.
            actual_choice (int): The user's actual choice.
        
        Returns:
            tuple: (new_state, reward, done) where:
                - new_state (int): The next situation index.
                - reward (int): 1 if the prediction was correct, -1 otherwise.
                - done (bool): Whether the last situation has been reached.
        """
        if action == actual_choice:
            reward = 1  # Reward for correct prediction
        else:
            reward = -1  # Penalty for incorrect prediction
            
        self.move_next()
        done = self.is_done()
        return self.x, reward, done

    def move_next(self):
        """Move to the next situation."""
        self.x += 1
        
    def is_done(self):
        """
        Check if we've reached the last situation.
        
        Returns:
            bool: True if this is the last situation, False otherwise.
        """
        return self.x == len(self.user_history) - 1
    
    def get_state(self):
        """
        Get the current situation index.
        
        Returns:
            int: The current situation index.
        """
        return self.x
    
    def reset(self):
        """Reset the environment to the first situation."""
        self.x = 0
        
        
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


def data_call(path):
    """
    Load user history and features data from CSV files.
    
    Args:
        path (str): Directory path containing the data files.
        
    Returns:
        tuple: (user_history_df, features_df) - DataFrames containing user history and features.
    """
    user_history_df = pd.read_csv(path + 'user_history.csv')
    features_df = pd.read_csv(path + 'features.csv')
    return user_history_df, features_df


def data(id, path, user):
    """
    Process data for a specific user.
    
    Extracts and preprocesses the data for a given user ID, converting it to
    the format required by the Environment and PolicyGradientAgent classes.
    
    Args:
        id (int): User ID.
        path (str): Directory path containing the data files.
        user (dict): Dictionary to store user data.
        
    Returns:
        tuple: (user_history, features, num_situations, feature_dim, feature_names) where:
            - user_history (list): List of the user's choices.
            - features (ndarray): 3D array of shape [num_situations, num_alternatives, feature_dim].
            - num_situations (int): Number of choice situations.
            - feature_dim (int): Dimensionality of the feature vectors.
            - feature_names (list): Names of features.
    """
    # Load data
    user_history_df, features_df = data_call(path)
    
    # Extract data for this user
    col1 = user_history_df.columns[1:]
    col2 = features_df.columns[1:]
    
    user.update({id: {"user_history_df": user_history_df.loc[user_history_df["id"] == id, col1],
                     "features_df": features_df.loc[features_df["id"] == id, col2]}})
    
    user_history_df = user[id]["user_history_df"]
    features_df = user[id]["features_df"]

    # Convert user history to list
    user_history = user_history_df['choice'].tolist()
    
    # Add situation column to features DataFrame
    # Assumes each situation has the same number of alternatives (3 in this case)
    features_df["situation"] = np.repeat(list(range(0, max(user_history_df["situation"]) + 1)), 3)
    
    # Get feature columns (excluding situation and alternative)
    feature_cols = [col for col in features_df.columns if col not in ['situation', 'alternative']]
    feature_names = feature_cols
    
    # Get dimensions
    num_situations = features_df['situation'].nunique()
    num_actions = features_df['alternative'].nunique()
    feature_dim = len(feature_cols)
    
    # Create features array
    features = np.zeros((num_situations, num_actions, feature_dim))
    for _, row in features_df.iterrows():
        situation = int(row['situation'])
        alternative = int(row['alternative'])
        features[situation, alternative] = row[feature_cols]
        
    return user_history, features, num_situations, feature_dim, feature_names

def create_interactive_dashboard(user_dict, output_file='dashboard.html'):
    """
    Create an interactive HTML dashboard summarizing model performance.
    
    This function generates an HTML file with interactive visualizations of:
    - Overall accuracy distribution
    - Feature importance across users
    - Learning curves for selected users
    
    Args:
        user_dict (dict): Dictionary containing user data and model results.
        output_file (str): Path to save the HTML dashboard.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # Sample users if there are too many
    if len(user_dict) > 10:
        sampled_user_ids = sorted(list(user_dict.keys()))[:10]
    else:
        sampled_user_ids = sorted(list(user_dict.keys()))
        
    # Create the HTML file
    with open(output_file, 'w') as f:
        # HTML header
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reinforcement Learning Model Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { background: #f9f9f9; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1, h2 { color: #333; }
                .flex-container { display: flex; flex-wrap: wrap; }
                .flex-item { flex: 1; min-width: 400px; margin: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Reinforcement Learning Model Dashboard</h1>
        """)
        
        # Summary statistics
        accuracies = [user_data['accuracy'] for user_data in user_dict.values()]
        f.write(f"""
                <div class="card">
                    <h2>Summary Statistics</h2>
                    <p>Total Users: {len(user_dict)}</p>
                    <p>Average Accuracy: {np.mean(accuracies):.2%}</p>
                    <p>Minimum Accuracy: {np.min(accuracies):.2%}</p>
                    <p>Maximum Accuracy: {np.max(accuracies):.2%}</p>
                </div>
        """)
        
        # Accuracy distribution
        fig = px.histogram(
            x=accuracies, 
            nbins=20, 
            labels={'x': 'Accuracy'},
            title='Distribution of Prediction Accuracy Across Users'
        )
        fig.add_vline(x=np.mean(accuracies), line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {np.mean(accuracies):.2%}")
        
        f.write("""
                <div class="card">
                    <h2>Accuracy Distribution</h2>
                    <div id="accuracy-hist"></div>
                </div>
                <script>
                    var histData = 
        """)
        f.write(fig.to_json())
        f.write("""
                    ;
                    Plotly.newPlot('accuracy-hist', histData.data, histData.layout);
                </script>
        """)
        
        # Feature importance (if available)
        if 'weights' in user_dict[sampled_user_ids[0]]:
            # Get feature names
            feature_names = None
            if hasattr(user_dict[sampled_user_ids[0]]['agent'], 'feature_names'):
                feature_names = user_dict[sampled_user_ids[0]]['agent'].feature_names
            
            if not feature_names:
                feature_names = [f'Feature {i}' for i in range(len(user_dict[sampled_user_ids[0]]['weights']))]
            
            # Calculate average feature importance
            all_weights = np.array([user_dict[uid]['weights'] for uid in sampled_user_ids])
            avg_weights = np.mean(all_weights, axis=0)
            
            # Create feature importance bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=feature_names,
                    y=avg_weights,
                    marker_color=['red' if w < 0 else 'blue' for w in avg_weights]
                )
            ])
            fig.update_layout(
                title="Average Feature Importance",
                xaxis_title="Features",
                yaxis_title="Average Weight",
                xaxis={'categoryorder': 'total descending'}
            )
            
            f.write("""
                    <div class="card">
                        <h2>Feature Importance</h2>
                        <div id="feature-importance"></div>
                    </div>
                    <script>
                        var featureData = 
            """)
            f.write(fig.to_json())
            f.write("""
                        ;
                        Plotly.newPlot('feature-importance', featureData.data, featureData.layout);
                    </script>
            """)
        
        # Learning curves for sampled users
        f.write("""
                <div class="card">
                    <h2>Learning Curves</h2>
                    <div class="flex-container">
        """)
        
        for uid in sampled_user_ids:
            if 'agent' in user_dict[uid] and hasattr(user_dict[uid]['agent'], 'accuracy_history'):
                agent = user_dict[uid]['agent']
                
                # Create subplot with accuracy and reward
                fig = make_subplots(rows=2, cols=1, 
                                   subplot_titles=("Accuracy per Episode", "Reward per Episode"))
                
                # Add accuracy trace
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(agent.accuracy_history) + 1)),
                        y=agent.accuracy_history,
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='green')
                    ),
                    row=1, col=1
                )
                
                # Add reward trace
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(agent.reward_history) + 1)),
                        y=agent.reward_history,
                        mode='lines+markers',
                        name='Reward',
                        line=dict(color='blue')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=400, width=500, title_text=f"User {uid} Learning Progress")
                
                f.write(f"""
                        <div class="flex-item">
                            <div id="learning-user-{uid}"></div>
                            <script>
                                var learningData{uid} = 
                """)
                f.write(fig.to_json())
                f.write(f"""
                                ;
                                Plotly.newPlot('learning-user-{uid}', learningData{uid}.data, learningData{uid}.layout);
                            </script>
                        </div>
                """)
        
        f.write("""
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)
    
    print(f"Interactive dashboard created at {output_file}")

def visualize_overall_performance(user_dict, save_dir=None):
    """
    Visualize the overall model performance across all users.
    
    Args:
        user_dict (dict): Dictionary containing user data, including accuracy.
        save_dir (str, optional): Directory to save figures. If None, figures are displayed.
    """
    # Create directory if it doesn't exist
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Extract accuracy values
    accuracies = [user_data['accuracy'] for user_data in user_dict.values()]
    
    # Histogram of accuracies
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(accuracies), color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean accuracy: {np.mean(accuracies):.2f}')
    plt.title('Distribution of Prediction Accuracy Across Users')
    plt.xlabel('Prediction Accuracy')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'accuracy_distribution.png'))
        plt.close()
    else:
        plt.show()
    
    # Feature importance across users (aggregated weights)
    if 'weights' in user_dict[list(user_dict.keys())[0]]:
        # Get feature names from the first user (assuming all users have the same features)
        agent = user_dict[list(user_dict.keys())[0]]['agent']
        if hasattr(agent, 'feature_names') and agent.feature_names:
            feature_names = agent.feature_names
            
            # Get average weights across all users
            all_weights = np.array([user_data['weights'] for user_data in user_dict.values()])
            avg_weights = np.mean(all_weights, axis=0)
            std_weights = np.std(all_weights, axis=0)
            
            # Sort by absolute weight value
            sorted_indices = np.argsort(np.abs(avg_weights))[::-1]
            sorted_avg_weights = avg_weights[sorted_indices]
            sorted_std_weights = std_weights[sorted_indices]
            sorted_feature_names = [feature_names[i] for i in sorted_indices]
            
            # Plot average feature weights
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(sorted_avg_weights)), sorted_avg_weights, yerr=sorted_std_weights,
                         capsize=5, alpha=0.7, color='lightblue', edgecolor='black')
            
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.xticks(range(len(sorted_avg_weights)), sorted_feature_names, rotation=45, ha='right')
            plt.title('Average Feature Importance Across All Users')
            plt.ylabel('Average Weight Value')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'average_feature_importance.png'))
                plt.close()
            else:
                plt.show()


def main(user_history, features, feature_dim, feature_names=None, learned_weights=None, 
         visualize=True, save_dir=None):
    """
    Main training function for the policy gradient agent.
    
    Trains an agent to predict user choices based on historical data.
    
    Args:
        user_history (list): List of the user's choices.
        features (ndarray): 3D array of shape [num_situations, num_alternatives, feature_dim].
        feature_dim (int): Dimensionality of the feature vectors.
        feature_names (list, optional): Names of features. Defaults to None.
        learned_weights (ndarray, optional): Initial weights for the agent. Defaults to None.
        visualize (bool, optional): Whether to create visualizations. Defaults to True.
        save_dir (str, optional): Directory to save visualizations. Defaults to None.
        
    Returns:
        tuple: (agent, env, weights) containing the trained agent, environment, and learned weights.
    """
    # Create visualization directory if needed
    if visualize and save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Hyperparameters
    alpha = 0.001  # Learning rate
    beta = 1.0     # Softmax temperature
    gamma = 0.95   # Discount factor
    num_episodes = 100  # Number of training episodes
    epsilon = 0.05  # Exploration rate
    
    # Initialize environment and agent
    env = Environment(user_history, features)
    agent = PolicyGradientAgent(alpha, beta, feature_dim, epsilon, gamma)
    
    # Set feature names if provided
    if feature_names:
        agent.set_feature_names(feature_names)
    
    # Use learned weights if provided
    if learned_weights is not None:
        agent.weights = learned_weights
        
    # Training loop
    for k in range(num_episodes):
        env.reset()
        done = False
        history = []
        total_reward = 0
        
        # Episode loop
        while not done:
            actual_choice = env.get_actual_choice()
            action, action_probabilities = agent.select_action(env.get_state(), env.features)
            state, reward, done = env.step(action, actual_choice) 
            history.append((action, state, reward, action_probabilities))
            total_reward += reward
        
        # Update policy based on this episode
        agent.update_policy(history, features)
        
        # Print progress every 10 episodes
        if (k + 1) % 10 == 0:
            accuracy = agent.accuracy_history[-1]
            print(f"Episode {k+1}/{num_episodes} - Reward: {total_reward:.1f}, Accuracy: {accuracy:.2%}")
    
    # Create visualizations if requested
    if visualize:
        # Visualize learning progress
        if save_dir:
            progress_path = os.path.join(save_dir, 'learning_progress.png')
            agent.visualize_learning_progress(save_path=progress_path)
            print(f"Learning progress visualization saved to {progress_path}")
            
            # Visualize final weights
            weights_path = os.path.join(save_dir, 'feature_weights.png')
            agent.visualize_weights(save_path=weights_path)
            print(f"Feature weights visualization saved to {weights_path}")
            
            # Create weight evolution animation
            animation_path = os.path.join(save_dir, 'weight_evolution.gif')
            agent.create_weight_evolution_animation(save_path=animation_path)
            print(f"Weight evolution animation saved to {animation_path}")
            
            # Visualize choice probabilities
            probs_path = os.path.join(save_dir, 'choice_probabilities.png')
            agent.visualize_choice_probabilities(features, user_history, save_path=probs_path)
            print(f"Choice probabilities visualization saved to {probs_path}")
        else:
            # Display visualizations if no save path provided
            agent.visualize_learning_progress()
            agent.visualize_weights()
            agent.create_weight_evolution_animation()
            agent.visualize_choice_probabilities(features, user_history)

    # Print final weights
    agent.print_weights()
    
    return agent, env, agent.weights


if __name__ == "__main__":
    """
    Main script execution.
    
    Loads data for each user, trains an agent, evaluates its performance,
    and reports accuracy metrics with visualizations.
    """
    total_correct_predictions = 0
    user = dict()
    path = "./"  # Data directory path - update as needed
    
    # Create visualization directory
    vis_dir = "visualization_results"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Process each user
    for id in list(range(1, 501)):
        print(f"\n--- Training model for User {id} ---")
        
        # Create user-specific visualization directory
        user_vis_dir = os.path.join(vis_dir, f"user_{id}")
        
        # Load and preprocess data
        user_history, features, num_situations, feature_dim, feature_names = data(id, path, user)
        
        # Train the agent (using existing weights if available)
        if "agent" in user[id].keys():
            trained_agent, trained_env, weights = main(
                user_history, features, feature_dim, feature_names,
                user[id]["agent"].weights, visualize=True, save_dir=user_vis_dir
            )
        else:
            trained_agent, trained_env, weights = main(
                user_history, features, feature_dim, feature_names,
                visualize=True, save_dir=user_vis_dir
            )
            
        # Evaluate the trained agent
        num_eval_episodes = 1
        correct_predictions = 0

        for episode in range(num_eval_episodes):
            for situation in range(num_situations):
                actual_choice = user_history[situation]
                action, action_probabilities = trained_agent.select_action(situation, features)
                
                # Count correct predictions
                if action == actual_choice:
                    correct_predictions += 1
                    total_correct_predictions += 1
                    
        # Calculate and report accuracy for this user
        accuracy = correct_predictions / (num_eval_episodes * num_situations)
        print(f"User {id} accuracy: {accuracy:.2%}")
        
        # Store results
        user[id].update({
            "env": trained_env,
            "agent": trained_agent, 
            "accuracy": accuracy, 
            "weights": weights
        })
        
        # Only process a few users if testing
        if id >= 50:  # Comment this out to process all users
            break
    
    # Calculate and report overall accuracy
    total_users = id
    accuracy = total_correct_predictions / (num_eval_episodes * num_situations * total_users)
    print(f"\nOverall prediction accuracy: {accuracy:.2%}")
    
    # Create overall performance visualizations
    visualize_overall_performance(user, save_dir=vis_dir)
    # Create overall performance visualizations
    visualize_overall_performance(user, save_dir=vis_dir)
    print(f"\nOverall performance visualizations saved to {vis_dir}")
    
    
    
    # Extract feature importance from all users
    all_user_weights = []
    all_user_ids = []
    
    for user_id, user_data in user.items():
        if 'weights' in user_data:
            all_user_weights.append(user_data['weights'])
            all_user_ids.append(user_id)
    
    all_user_weights = np.array(all_user_weights)
    
    # Create a heatmap of feature importance across users
    if len(all_user_weights) > 0:
        plt.figure(figsize=(14, 10))
        
        # Get feature names from the first user
        feature_names = user[all_user_ids[0]]['agent'].feature_names if hasattr(user[all_user_ids[0]]['agent'], 'feature_names') else None
        
        if feature_names:
            # Create a heatmap of weights
            sns.heatmap(all_user_weights, cmap='coolwarm', center=0, 
                        xticklabels=feature_names, yticklabels=all_user_ids,
                        annot=False)
            plt.title('Feature Importance Across Users')
            plt.xlabel('Features')
            plt.ylabel('User ID')
            plt.tight_layout()
            
            heatmap_path = os.path.join(vis_dir, 'user_feature_importance_heatmap.png')
            plt.savefig(heatmap_path)
            plt.close()
            print(f"User feature importance heatmap saved to {heatmap_path}")
            
            # Create a cluster map to see user similarity based on weights
            plt.figure(figsize=(14, 10))
            
            # Check for and handle non-finite values before creating the clustermap
            if np.any(~np.isfinite(all_user_weights)):
                print("Warning: Non-finite values found in weights. Replacing with zeros for clustering.")
                # Create a copy to avoid modifying the original data
                clustering_weights = np.copy(all_user_weights)
                # Replace non-finite values with zeros
                clustering_weights[~np.isfinite(clustering_weights)] = 0
            else:
                clustering_weights = all_user_weights
            
            try:
                cluster_grid = sns.clustermap(clustering_weights, cmap='coolwarm', center=0,
                                             xticklabels=feature_names, yticklabels=all_user_ids,
                                             standard_scale=1, method='ward')
                plt.title('User Clusters Based on Feature Importance')
                
                cluster_path = os.path.join(vis_dir, 'user_clusters.png')
                plt.savefig(cluster_path)
                plt.close()
                print(f"User clustering visualization saved to {cluster_path}")
            except Exception as e:
                print(f"Could not create cluster map: {str(e)}")
                plt.close()
            
    # Create visualizations comparing accuracy to model parameters
    if len(user) > 1:
        # Extract accuracies
        user_ids = list(user.keys())
        accuracies = [user[uid]['accuracy'] for uid in user_ids]
        
        # Plot accuracy vs user ID
        plt.figure(figsize=(12, 6))
        plt.bar(user_ids, accuracies, alpha=0.7, color='skyblue')
        plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                    label=f'Mean accuracy: {np.mean(accuracies):.2f}')
        plt.title('Prediction Accuracy per User')
        plt.xlabel('User ID')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        acc_path = os.path.join(vis_dir, 'accuracy_per_user.png')
        plt.savefig(acc_path)
        plt.close()
        print(f"Accuracy per user visualization saved to {acc_path}")
        
    print("\nTraining and visualization complete!")


    # After all other visualizations are complete:
    dashboard_path = os.path.join(vis_dir, 'model_dashboard.html')
    try:
        create_interactive_dashboard(user)
    except ImportError:
        print("Could not create interactive dashboard. plotly package may be missing.")
        print("Install with: pip install plotly")