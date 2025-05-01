from maze import MazeSetup
from RandomM import RandomModel
from learningAlgo import QLearningAgent
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import pandas as pd
import numpy as np
import pygame


def path_plot(env, path, title):
    """plot the agent path"""

    plt.figure(figsize=(8, 8))


    for y in range(env.rows):
        for x in range(env.col):
            if env.maze[y, x] == 1:
                plt.scatter(x, y, marker='x', color='black')  # Wall
            elif env.maze[y, x] == 2:
                plt.scatter(x, y, marker='x', color='green')  # Start
            elif env.maze[y, x] == 3:
                plt.scatter(x, y, marker='x', color='blue')   # Sub-goal
            elif env.maze[y, x] == 4:
                plt.scatter(x, y, marker='x', color='red')    # End goal


    if len(path) > 0:
        path_y, path_x = zip(*path)
        plt.plot(path_x, path_y, marker='o', color='yellow')
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


env = MazeSetup()

qL_params = {
    'alpha': 0.5,
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.1,
    'num_episodes': 500,
    'max_steps': 300
}


agent = QLearningAgent(
    rows= env.rows,
    cols= env.col,
    num_actions=env.action_space.n,
    alpha= qL_params['alpha'],
    gamma= qL_params['gamma'],
    epsilon= qL_params['epsilon'],
    epsilon_decay=qL_params['epsilon_decay'],
    epsilon_min= qL_params['epsilon_min']
)



total_ep= qL_params['num_episodes']
maxStep = qL_params['max_steps']
total_R= []
epsilon_ep = []   # Epsilon values
achievements = []   #achievements per episode
step_ep= []
exploration_act= []
exploitation_act = []
paths_ep ={}      #paths in episodes
visit_counter= np.zeros((env.rows, env.col))
max_q_ep = []


with open('QParams.json', 'w') as f:
    json.dump(qL_params, f, indent=4)


for episode in range(total_ep):
    state_pos= env.reset()

    completed= False
    tot_R = 0
    step= 0
    achieved_ = 0
    path= []
    exploration_ =0
    exploitation_ = 0

    while not completed and step < maxStep:
        y, x = state_pos
        visit_counter[y, x] += 1


        if np.random.rand() < agent.epsilon:
            action= np.random.choice(agent.num_actions)
            exploration_ += 1
        else:
            action = np.argmax(agent.q_table[y, x, :])
            exploitation_ += 1


        path.append(state_pos)
        # new step
        next_s, reward, completed, _ = env.step(action)

        # Update q table
        agent.update_q_value(state_pos, action, reward, next_s, completed)


        state_pos= next_s

        tot_R += reward

        if completed and reward == 60:
            achieved_ = 1

        if episode % 50 == 0:
            env.render(path=path)
            pygame.time.wait(150)  #wait for rendering so that gui not freeze

        step += 1

    total_R.append(tot_R)
    epsilon_ep.append(agent.epsilon)
    achievements.append(achieved_)
    step_ep.append(step)
    exploration_act.append(exploration_)
    exploitation_act.append(exploitation_)
    max_q_ep.append(np.max(agent.q_table))


    if episode in [0, int(total_ep / 2), total_ep - 1]:
        paths_ep[episode]= path.copy()


    agent.decay_epsilon()

    print(f"episode {episode + 1}/{total_ep}, total Reward: {tot_R}, "
          f"epsilon: {agent.epsilon:.3f}, achieved: {achieved_}")


data= {
    'Episode': range(1, total_ep + 1),
    'Total Reward': total_R,
    'Epsilon': epsilon_ep,
    'Steps': step_ep,
    'Success': achievements,
    'Exploration': exploration_act,
    'Exploitation': exploitation_act,
    'Max Q-value': max_q_ep
}
df = pd.DataFrame(data)
df.to_csv('training_metrics.csv', index=False)


qTable_s = agent.q_table.reshape(-1, agent.num_actions)

df_q_table = pd.DataFrame(qTable_s)
df_q_table.to_csv('q_table.csv', index=False)

random_agent = RandomModel(num_actions=env.action_space.n)


num_random_episodes = qL_params['num_episodes']

random_total_rewards = []
random_achievements = []
random_steps_per_episode = []


for episode in range(num_random_episodes):
    state_pos = env.reset()
    done = False
    total_reward = 0
    steps = 0
    achieved = 0
    path = []

    while not done and steps < maxStep:
        action = random_agent.choose_action(state_pos)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        path.append(state_pos)
        state_pos = next_state
        steps += 1

        if done and reward == 60:
            achieved = 1

    random_total_rewards.append(total_reward)
    random_achievements.append(achieved)
    random_steps_per_episode.append(steps)

random_data = {
    'Episode': range(1, num_random_episodes + 1),
    'Total Reward': random_total_rewards,
    'Steps': random_steps_per_episode,
    'Success': random_achievements,
}

random_training = pd.DataFrame(random_data)
random_training.to_csv('random_agent_metrics.csv', index=False)



sizeW = 10
rewards_s = pd.Series(total_R).rolling(sizeW).mean()

plt.figure(figsize=(12, 6))
plt.plot(range(total_ep), total_R, label='Total rewards per episode', alpha=0.3)
plt.plot(range(total_ep), rewards_s, label=f'{sizeW}-Episode Moving average', color='red')
plt.xlabel('episode')
plt.ylabel('Rewards')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve.png')
plt.show()


cumulativeAchievement = np.cumsum(achievements) / np.arange(1, total_ep + 1)
plt.figure(figsize=(12, 6))
plt.plot(range(1, total_ep + 1), cumulativeAchievement)
plt.xlabel('episode')
plt.ylabel('Cumulative achievement rate')
plt.title('Cumulative achievement')
plt.grid(True)
plt.savefig('cumulative_success_rate.png')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(range(total_ep), epsilon_ep)
plt.xlabel('Episode')
plt.ylabel('epsilon')
plt.title('Epsilon Decay')
plt.grid(True)
plt.savefig('epsilon_decay.png')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(range(total_ep), exploration_act, label='Exploratory Actions')
plt.plot(range(total_ep), exploitation_act, label='Exploitative Actions')
plt.xlabel('episode')
plt.ylabel('total actions')
plt.title('exploration vs. Exploitation  in each episode')
plt.legend()
plt.grid(True)
plt.savefig('exploration_exploitation.png')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(range(total_ep), max_q_ep)
plt.xlabel('episode')
plt.ylabel('max Q value')
plt.title('Q value convergence')
plt.grid(True)
plt.savefig('q_value_convergence.png')
plt.show()


# credit: book: Practical reinforcement learning with python
policy= np.argmax(agent.q_table, axis=2)
plt.figure(figsize=(8, 8))
sdf= plt.gca()
for y in range(policy.shape[0]):
    for x in range(policy.shape[1]):
        action = policy[y, x]
        if env.maze[y, x] != 1:
            dx, dy= env._action_to_direction[action][1], -env._action_to_direction[action][0]
            sdf.arrow(x, y, dx * 0.3, dy * 0.3, head_width=0.2, head_length=0.2, fc='k', ec='k')


for y in range(env.rows):
    for x in range(env.col):
        if env.maze[y, x] == 1:
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='gray')
            sdf.add_patch(rect)
        elif env.maze[y, x] == 2:
            plt.scatter(x, y, marker='s', color='green', s=200)  #start goal
        elif env.maze[y, x] == 3:
            plt.scatter(x, y, marker='s', color='blue', s=200)   #subgoal
        elif env.maze[y, x] == 4:
            plt.scatter(x, y, marker='s', color='red', s=200)    # End goal

plt.xlim(-0.5, env.col - 0.5)
plt.ylim(env.rows - 0.5, -0.5)
plt.title('Policy visualization')
plt.xlabel('Column')
plt.ylabel('Row')
plt.grid(True)
plt.savefig('policy_arrows.png')
plt.show()


states_Val = np.max(agent.q_table, axis=2)
plt.figure(figsize=(8, 8))
plt.imshow(states_Val, cmap='hot', interpolation='nearest')
plt.colorbar(label='State Value')
plt.title('state value Function heatmap')
plt.xlabel('Column')
plt.ylabel('Row')
plt.gca().invert_yaxis()
plt.savefig('state_value_heatmap.png')
plt.show()


plt.figure(figsize=(8, 8))
plt.imshow(visit_counter, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Visit Count')
plt.title('State Visit counter heatmap')
plt.xlabel('Column')
plt.ylabel('Row')
plt.gca().invert_yaxis()
plt.savefig('visit_count_heatmap.png')
plt.show()


for episode, path in paths_ep.items():
    path_plot(env, path, f'Agent Path at Episode {episode}')


alpha_training = [0.1, 0.5, 0.9]
results = {}
for alpha in alpha_training:
    print(f"\nTraining with alpha = {alpha}")
    qL_params['alpha'] = alpha

    agent = QLearningAgent(
        rows=env.rows,
        cols=env.col,
        num_actions=env.action_space.n,
        alpha=alpha,
        gamma=qL_params['gamma'],
        epsilon=qL_params['epsilon'],
        epsilon_decay=qL_params['epsilon_decay'],
        epsilon_min=qL_params['epsilon_min']
    )
    sum_R_alpha = []
    for episode in range(total_ep):
        state_pos= env.reset()
        completed= False
        tot_R = 0
        step= 0

        while not completed and step < maxStep:
            y, x = state_pos

            # epsilon greedy metod
            if np.random.rand() < agent.epsilon:
                action= np.random.choice(agent.num_actions)
            else:
                action = np.argmax(agent.q_table[y, x, :])

            # new step
            next_s, reward, completed, _ = env.step(action)
            #update q table
            agent.update_q_value(state_pos, action, reward, next_s, completed)
            # state update
            state_pos= next_s
            # total reward incremented
            tot_R += reward

            step += 1
        # Decay epsilon
        agent.decay_epsilon()
        # save total reward
        sum_R_alpha.append(tot_R)
    # finally save in dict for each alpha
    results[alpha]= sum_R_alpha

#plot different aphas effect on total reward
plt.figure(figsize=(12, 6))
sizeW = 10
for alpha, rewards in results.items():
    rewards_s = pd.Series(rewards).rolling(sizeW).mean()
    plt.plot(range(total_ep), rewards_s, label=f'Alpha = {alpha}')
plt.xlabel('episode')
plt.ylabel('Total reward')
plt.title('Effect of Alpha on total rewards')
plt.legend()
plt.grid(True)
plt.savefig('hyperparameter_analysis_alpha.png')
plt.show()

# Comparing Q learning Agent and Random Agent Performance
# Total Rewards Comparison
plt.figure(figsize=(12, 6))
plt.plot(range(total_ep), total_R, label='Q-learning Agent')
plt.plot(range(num_random_episodes), random_total_rewards, label='Random Agent', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards Comparison')
plt.legend()
plt.grid(True)
plt.savefig('total_rewards_comparison.png')
plt.show()


q_learning_cumulative_success = np.cumsum(achievements) / np.arange(1, total_ep + 1)
random_cumulative_success = np.cumsum(random_achievements) / np.arange(1, num_random_episodes + 1)

plt.figure(figsize=(12, 6))
plt.plot(range(1, total_ep + 1), q_learning_cumulative_success, label='Q-learning Agent')
plt.plot(range(1, num_random_episodes + 1), random_cumulative_success, label='Random Agent', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Cumulative Success Rate')
plt.title('Cumulative Success Rate Comparison')
plt.legend()
plt.grid(True)
plt.savefig('cumulative_success_rate_comparison.png')
plt.show()
#close environment
pygame.quit()






