import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
%matplotlib inline

def plot_values(greedy_reward, epsilon_reward, greedy_accuracy, epsilon_accuracy):
  fig = plt.figure(figsize=[30,8])
  ax2 = fig.add_subplot(121)
  ax2.plot(greedy_reward, label='Greedy Cumulative Average Rewards')
  ax2.plot(epsilon_reward, label='Epsilon Cumulative Average Rewards')
  ax2.set_xlabel('Timestep')
  ax2.set_ylabel('Average Reward')
  ax2.legend()
  
  ax3 = fig.add_subplot(122)
  ax3.plot(greedy_accuracy, label='Greedy Percent Optimal Actions')
  ax3.plot(epsilon_accuracy, label='Epsilon Percent Optimal Actions')
  ax3.set_xlabel('Timestep')
  ax3.set_ylabel('% Optimal Action')
  ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
  ax3.legend()
  
 
 class ArmedBandit():
  def __init__(self, num_arms, time_steps, epsilon=0.0):
    self.num_arms = num_arms
    self.time_steps = time_steps
    self.epsilon = epsilon

  def pick_arm(self):
    q_values = [0] * self.num_arms
    arm_rewards = [0] * self.num_arms
    arm_counts = [0] * self.num_arms
    rewards = []
    cumulative_rewards = []
    average_correct = []
    individual_rewards = []
    num_correct = 0

    # Reward for each arm is randomly picked with normal distribution
    for i in range(self.num_arms):
      individual_rewards.append(np.random.normal(0,1,1))

    optimal_arm = np.argmax(individual_rewards)

    for i in range(1,self.time_steps+1):
      # Pick which arm to pull
      if np.random.random() < self.epsilon:
        arm = np.random.choice(self.num_arms) 
      else:
        arm = np.argmax(q_values)
      #reward = individual_rewards[arm]
      if arm == optimal_arm:
        num_correct += 1
      
      # Add up reward and update q_values
      reward = individual_rewards[arm]
      arm_rewards[arm] += reward
      arm_counts[arm] += 1
      q_values[arm] = q_values[arm] + (1/arm_counts[arm])*(reward - q_values[arm])
      
      # Collect cumulative reward and optimal pick for each timestamp
      average_correct.append(num_correct / i)
      rewards.append(reward)
      cumulative_rewards.append(sum(rewards) / len(rewards))
    
    return cumulative_rewards, average_correct
 
'''
Graph on the left averages cumulative rewards for each timestamp
Graph on the right averages cumulative optimal picks for each timestamp
'''
# Model Parameters
num_runs = 200
timesteps = 100
bandits = 5
epsilon_ratio = 0.4

# Data collection variables
greedy_cum_reward = [0] * timesteps
epsilon_cum_reward = [0] * timesteps
epsilon_optimal = [0] * timesteps
greedy_optimal = [0] * timesteps

for i in range(num_runs):
  greedy = ArmedBandit(bandits, timesteps, 0)
  epsilon = ArmedBandit(bandits,timesteps, epsilon_ratio)

  # Returns cumulative rewards and optimal pickinfor each timestep
  greedy_rewards, greedy_correct = greedy.pick_arm()
  epsilon_rewards, epsilon_correct = epsilon.pick_arm()
  
  # Average every time step for each iteration
  for j in range(len(greedy_rewards)):
    greedy_cum_reward[j] += greedy_rewards[j]/num_runs
    epsilon_cum_reward[j] += epsilon_rewards[j]/num_runs
    
    greedy_optimal[j] += (greedy_correct[j]/num_runs)*100
    epsilon_optimal[j] += (epsilon_correct[j]/num_runs)*100


plot_values(greedy_cum_reward, epsilon_cum_reward, greedy_optimal, epsilon_optimal)


