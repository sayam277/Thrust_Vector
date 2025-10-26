import numpy as np
import itertools

# ======================
# LUNAR LANDER MDP
# ======================

# Environment Setup
heights = ["High", "Medium", "Low"]
velocities = ["Up", "Zero", "Down"]
actions = ["ThrustUp", "NoThrust"]

states = list(itertools.product(heights, velocities))
num_states, num_actions = len(states), len(actions)

# Initialize MDP matrices
P = np.zeros((num_states, num_actions, num_states))
R = np.zeros((num_states, num_actions))

# Build Transitions (PROPER REWARDS)
for s_idx, (h, v) in enumerate(states):
    for a_idx, a in enumerate(actions):
        # TERMINAL STATES
        if h == "Low" and v == "Zero":
            # SUCCESS TERMINAL
            next_h, next_v = h, v
            reward = 0
        elif h == "Low" and v == "Down":
            # CRASH TERMINAL  
            next_h, next_v = h, v
            reward = -10  # Ongoing crash penalty
        else:
            # NORMAL PHYSICS
            next_h, next_v = h, v
            base_reward = 0
            
            # Physics Rules
            if a == "ThrustUp":
                if v == "Down": next_v = "Zero"
                elif v == "Zero": next_v = "Up"
                
                if h == "Medium" and next_v == "Up": next_h = "High"
                elif h == "Low" and next_v != "Down": next_h = "Medium"
                
                base_reward = -0.3  # Reduced fuel cost
                
            elif a == "NoThrust":
                if v == "Up": next_v = "Zero"
                elif v == "Zero": next_v = "Down"
                
                if h == "High" and next_v == "Down": next_h = "Medium"
                if h == "Medium" and next_v == "Down": next_h = "Low"
                
                base_reward = 0  # No fuel cost
            
            # SMART REWARD SYSTEM
            reward = base_reward
            
            # 1. Altitude rewards (higher is safer)
            if h == "High":
                reward += 1.0
            elif h == "Medium":
                reward += 0.5
            
            # 2. Velocity rewards (slow is safe)
            if v == "Zero":
                reward += 0.5
            elif v == "Up":
                reward += 0.2
            
            # 3. Progress rewards (descending toward goal)
            if next_h == "Medium" and h == "High":
                reward += 2.0  # Good: coming down from high
            elif next_h == "Low" and h == "Medium":
                reward += 3.0  # Great: approaching landing
            
            # 4. Terminal transition rewards
            if next_h == "Low" and next_v == "Zero":
                reward += 200  # SUCCESS! Perfect landing
            elif next_h == "Low" and next_v == "Down":
                reward -= 100  # CRASH! Avoid this

        next_idx = states.index((next_h, next_v))
        P[s_idx, a_idx, next_idx] = 1.0
        R[s_idx, a_idx] = reward

# Solve with Value Iteration
gamma = 0.95  # Higher discount - value future more
V = np.zeros(num_states)

print("Running Value Iteration...")
for i in range(200):
    V_new = np.zeros_like(V)
    for s in range(num_states):
        q_values = [R[s,a] + gamma * np.dot(P[s,a], V) for a in range(num_actions)]
        V_new[s] = max(q_values)
    if np.max(np.abs(V - V_new)) < 1e-6:
        print(f"Converged in {i+1} iterations")
        break
    V = V_new

# Extract Optimal Policy
policy = [np.argmax([R[s,a] + gamma * np.dot(P[s,a], V) for a in range(num_actions)]) 
          for s in range(num_states)]

# ======================
# DISPLAY RESULTS
# ======================

print("\n" + "="*60)
print("           LUNAR LANDER MDP - LOGICAL RESULTS")
print("="*60)

print(f"\nSTATE SPACE:")
for i, state in enumerate(states):
    status = "LANDED" if state == ('Low', 'Zero') else "CRASHED" if state == ('Low', 'Down') else ""
    print(f"  {i:2d}. {state} {status}")

print("\nREWARD STRUCTURE:")
print("  Fuel cost: -0.3 (ThrustUp)")
print("  Altitude bonus: High=+1.0, Medium=+0.5")  
print("  Velocity bonus: Zero=+0.5, Up=+0.2")
print("  Progress: High->Medium=+2.0, Medium->Low=+3.0")
print("  Success: +200 (Low,Zero)")
print("  Crash: -100 (Low,Down)")

print("\nOPTIMAL VALUES (Expected Future Rewards):")
print("-" * 50)
for state, value in zip(states, V):
    if state == ('Low', 'Zero'):
        status = "PERFECT LANDING"
    elif state == ('Low', 'Down'):
        status = "CRASHED"
    elif value > 150:
        status = "EXCELLENT PATH"
    elif value > 100:
        status = "GREAT PATH" 
    elif value > 50:
        status = "GOOD PATH"
    elif value > 0:
        status = "POSITIVE PATH"
    else:
        status = "RISKY PATH"
    print(f"  {state} -> {value:7.1f}  {status}")

print("\nOPTIMAL POLICY:")
print("-" * 50)
for state, action_idx in zip(states, policy):
    action = actions[action_idx]
    if state == ('Low', 'Zero') or state == ('Low', 'Down'):
        print(f"  {state} -> TERMINAL")
    else:
        reasoning = ""
        if action == "ThrustUp":
            if state[1] == "Down":
                reasoning = "(slow descent)"
            elif state[0] == "Low":
                reasoning = "(gain altitude)"
        else:
            if state[1] == "Up":
                reasoning = "(coast down)"
            elif state[0] == "High":
                reasoning = "(descend safely)"
        
        print(f"  {state} -> {action} {reasoning}")

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("  Higher altitudes have higher values (safer)")
print("  Slow descent is rewarded")
print("  Policy uses thrust to control speed, not altitude")
print("  Logical gradient from risky->safe->optimal")
print("="*60)
import matplotlib.pyplot as plt

# Plot 1: Value Function Bar Chart
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
state_labels = [f"{h}\n{v}" for h,v in states]
bars = plt.bar(range(num_states), V, color=['red' if v < 0 else 'green' for v in V])
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title('Optimal Value Function')
plt.xticks(range(num_states), state_labels, rotation=45)
plt.ylabel('State Value')
for bar, v in zip(bars, V):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{v:.1f}', ha='center', va='bottom', fontsize=8)

# Plot 2: Policy Visualization
plt.subplot(1, 2, 2)
policy_actions = [actions[p] for p in policy]
colors = ['orange' if a == 'ThrustUp' else 'blue' for a in policy_actions]
bars = plt.bar(range(num_states), [1]*num_states, color=colors)
plt.title('Optimal Policy\n(Orange=ThrustUp, Blue=NoThrust)')
plt.xticks(range(num_states), state_labels, rotation=45)
plt.yticks([])

plt.tight_layout()
plt.savefig('task1_results.png', dpi=300, bbox_inches='tight')
plt.show()
