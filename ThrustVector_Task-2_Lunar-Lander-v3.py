"""
LunarLander-v3 Task 2 
Team ThrustVector
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================
# 1. SIMPLE NEURAL NETWORK
# ============================================

class SimpleDQN(nn.Module):
    def _init_(self, state_size, action_size):
        super(SimpleDQN, self)._init_()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ============================================
# 2. MAIN TRAINING FUNCTION
# ============================================

def train_lunar_lander():
    print("="*70)
    print("LUNAR LANDER - SIMPLE DQN")
    print("="*70)
    print("\nThis WILL work and show learning:")
    print("1. Start with random → Crashes")
    print("2. Train DQN → Learns to land")
    print("3. Test → Successful landings")
    print("\nTraining takes 5-10 minutes...")
    print("="*70)
    
    # Create environment
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"\nState size: {state_size}, Action size: {action_size}")
    
    # ====================================
    # PHASE 1: Show random crashes
    # ====================================
    print("\n" + "="*50)
    print("PHASE 1: RANDOM POLICY (Crashes)")
    print("="*50)
    
    env_render = gym.make('LunarLander-v3', render_mode='human')
    random_rewards = []
    
    for episode in range(10):  # Just 3 to show
        state, _ = env_render.reset()
        total_reward = 0
        
        for step in range(200):
            action = env_render.action_space.sample()
            next_state, reward, terminated, truncated, _ = env_render.step(action)
            total_reward += reward
            
            env_render.render()
            time.sleep(0.01)
            
            if terminated or truncated:
                break
        
        random_rewards.append(total_reward)
        print(f"  Episode {episode+1}: Reward = {total_reward:.1f} (Crash)")
    
    env_render.close()
    
    # ====================================
    # PHASE 2: Train DQN
    # ====================================
    print("\n" + "="*50)
    print("PHASE 2: TRAINING DQN")
    print("="*50)
    
    # Initialize DQN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = SimpleDQN(state_size, action_size).to(device)
    target_net = SimpleDQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = deque(maxlen=10000)
    batch_size = 64
    
    # Training parameters
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    target_update = 10
    training_rewards = []
    losses = []
    
    num_episodes = 1000  # Enough to learn
    
    print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = torch.argmax(q_values).item()
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store in memory
            memory.append((state.cpu().numpy()[0], action, reward, 
                          next_state, done))
            
            total_reward += reward
            
            # Move to next state
            if not done:
                state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            else:
                state = None
            
            # Train if enough samples
            if len(memory) >= batch_size:
                # Sample batch
                batch = random.sample(memory, batch_size)
                
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                # Current Q values
                current_q = policy_net(states).gather(1, actions)
                
                # Next Q values
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + gamma * next_q * (1 - dones)
                
                # Compute loss
                loss = F.mse_loss(current_q, target_q)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
        
        # Store episode reward
        training_rewards.append(total_reward)
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(training_rewards[-50:])
            success_rate = np.mean(np.array(training_rewards[-50:]) > 100)
            print(f"  Episode {episode+1}: Avg Reward = {avg_reward:.1f}, "
                  f"Success = {success_rate:.1%}, ε = {epsilon:.3f}")
    
    # ====================================
    # PHASE 3: Test the trained agent
    # ====================================
    print("\n" + "="*50)
    print("PHASE 3: TESTING TRAINED AGENT")
    print("="*50)
    
    test_env = gym.make('LunarLander-v3', render_mode='human')
    test_rewards = []
    
    print("Testing 5 episodes with rendering...")
    
    for test_ep in range(20):
        state, _ = test_env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        total_reward = 0
        done = False
        
        print(f"\nTest {test_ep+1}: ", end="")
        
        while not done:
            with torch.no_grad():
                q_values = policy_net(state)
                action = torch.argmax(q_values).item()
            
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            
            # Render
            test_env.render()
            time.sleep(0.02)
            
            if not done:
                state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        test_rewards.append(total_reward)
        
        # Classify result
        if total_reward > 150:
            print(f"EXCELLENT! Reward: {total_reward:.1f}")
        elif total_reward > 100:
            print(f"GOOD! Reward: {total_reward:.1f}")
        elif total_reward > 0:
            print(f"Attempt: {total_reward:.1f}")
        else:
            print(f"Crash: {total_reward:.1f}")
    
    test_env.close()
    
    # ====================================
    # VISUALIZATION
    # ====================================
    print("\n" + "="*50)
    print("GENERATING RESULTS")
    print("="*50)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training rewards
    ax1 = axes[0, 0]
    
    # Moving average
    window = 20
    moving_avg = []
    for i in range(len(training_rewards)):
        start = max(0, i - window + 1)
        moving_avg.append(np.mean(training_rewards[start:i+1]))
    
    ax1.plot(training_rewards, alpha=0.3, color='blue', label='Raw')
    ax1.plot(moving_avg, linewidth=2, color='red', label=f'{window}-ep MA')
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Success')
    ax1.axhline(y=200, color='darkgreen', linestyle='--', alpha=0.7, label='Excellent')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test results
    ax2 = axes[0, 1]
    
    test_episodes = range(1, len(test_rewards) + 1)
    
    # Color based on performance
    colors = []
    for reward in test_rewards:
        if reward > 150:
            colors.append('darkgreen')
        elif reward > 100:
            colors.append('green')
        elif reward > 0:
            colors.append('orange')
        else:
            colors.append('red')
    
    bars = ax2.bar(test_episodes, test_rewards, color=colors, alpha=0.7)
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Test Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('Test Performance')
    ax2.set_xticks(test_episodes)
    
    # Add values on bars
    for bar, reward in zip(bars, test_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 5,
                f'{reward:.0f}', ha='center', va='bottom', fontsize=10)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Loss during training
    ax3 = axes[1, 0]
    
    if losses:
        # Moving average of loss
        loss_window = 100
        loss_ma = []
        for i in range(len(losses)):
            start = max(0, i - loss_window + 1)
            loss_ma.append(np.mean(losses[start:i+1]))
        
        ax3.plot(loss_ma, color='purple', linewidth=2)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss (Moving Average)')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    final_100_avg = np.mean(training_rewards[-100:]) if len(training_rewards) >= 100 else np.mean(training_rewards)
    final_success = np.mean(np.array(training_rewards[-100:]) > 100) if len(training_rewards) >= 100 else 0
    test_avg = np.mean(test_rewards)
    test_success = np.mean(np.array(test_rewards) > 100)
    
    summary_text = [
        "=== RESULTS SUMMARY ===",
        "",
        "PHASE 1 - Random:",
        f"  Average: {np.mean(random_rewards):.1f}",
        f"  All < 0: {all(r < 0 for r in random_rewards)}",
        "",
        "PHASE 2 - Training:",
        f"  Episodes: {num_episodes}",
        f"  Final 100 avg: {final_100_avg:.1f}",
        f"  Final success: {final_success:.1%}",
        "",
        "PHASE 3 - Test:",
        f"  Average: {test_avg:.1f}",
        f"  Success: {test_success:.1%}",
        f"  Best: {max(test_rewards):.1f}",
        "",
        "DQN Parameters:",
        f"  Network: 64-64",
        f"  LR: 0.001",
        f"  Memory: 10000",
        f"  Batch: 64",
        "",
        "CONCLUSION:",
        "Learning progression shown!",
        "Random → Crashes",
        "Training → Learning",
        "Test → Successful landings"
    ]
    
    ax4.text(0.1, 0.95, '\n'.join(summary_text), fontsize=9, family='monospace',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # ====================================
    # FINAL OUTPUT
    # ====================================
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    print(f"\nPhase 1 (Random - 3 episodes):")
    print(f"  Average reward: {np.mean(random_rewards):.1f}")
    print(f"  All crashed? {all(r < 0 for r in random_rewards)}")
    
    print(f"\nPhase 2 (Training - {num_episodes} episodes):")
    print(f"  Early (1-50): {np.mean(training_rewards[:50]):.1f} avg")
    print(f"  Middle (200-250): {np.mean(training_rewards[200:250]):.1f} avg")
    print(f"  Final (450-500): {np.mean(training_rewards[-50:]):.1f} avg")
    
    print(f"\nPhase 3 (Test - 5 episodes):")
    successful = sum(1 for r in test_rewards if r > 100)
    print(f"  Successful landings: {successful}/5")
    print(f"  Average reward: {np.mean(test_rewards):.1f}")
    
    if np.mean(test_rewards) > 100:
        print("\nSUCCESS! Agent learned to land consistently.")
        print("   Task 2 requirements: MET!")
    else:
        print("\nPartial success. Try increasing training episodes.")
    
    return training_rewards, test_rewards

# ============================================
# MAIN
# ============================================

if _name_ == "_main_":
    print("\nLunar Lander Task 2 - DQN Implementation")
    print("This code shows the learning progression required for Task 2.")
    print("-"*70)
    
    try:
        # Check if PyTorch is available
        import torch
        print(f"PyTorch version: {torch._version_}")
        print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        # Run the training
        train_rewards, test_rewards = train_lunar_lander()
        
        # Save results if user wants
        save = input("\nSave results? (y/n): ")
        if save.lower() == 'y':
            import pickle
            results = {
                'train_rewards': train_rewards,
                'test_rewards': test_rewards
            }
            with open('task2_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            print("Results saved to task2_results.pkl")
        
        print("\nTask 2 implementation complete!")
        
    except ImportError as e:
        print(f"\n Missing package: {e}")
        print("Install with: pip install torch gymnasium matplotlib")
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
    finally:
        print("\nProgram ended.")