import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Memory():
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        # actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
            )
        
        #critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
            )
    
    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()
    
    def evaluate(self, state, action):
        # return the 1) log probability of the evaluated action and 2) state values to calculate the advantage, 3) entropy reward of the probabilistic policy.

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_values = self.critic(state)

        return action_logprobs, torch.squeeze(state_values), dist_entropy

class PPO():
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MSELoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimation
        rewards = []
        discounted_reward = 0
        
        # for several trajectories
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        # Normalizing rewards
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # optimizing policy for K epochs
        for _ in range(self.K_epochs):
            # evaluating old actions and state values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # finding surogate loss 
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = - torch.min(surr1, surr2) + 0.5*self.MSELoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def main():
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    solved_reward = 230
    log_interval = 20
    max_episodes = 50000
    max_timesteps = 300
    hidden_dim = 64
    update_timestep = 2000
    lr = 2e-3
    gamma = 0.99
    K_epochs =4
    eps_clip = 0.2
    random_seed = None

    if random_seed:
         torch.manual_seed(random_seed)
         env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, hidden_dim, lr, gamma, K_epochs, eps_clip)

    # logging variables and files
    running_reward = 0
    avg_length = 0
    timestep = 0
    log_f = open("log.txt", "w+")

    # training loop
    for epochs in range(1, max_episodes+1):
        state = env.reset()
        total_rewards_episode = 0
        for t in range(max_timesteps):
            timestep += 1
            with torch.no_grad():
                action = ppo.policy.act(state, memory)
            state, reward, done, _ = env.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            total_rewards_episode  += reward

            if render:
                env.render()
            if done:
                break

        running_reward += total_rewards_episode
        avg_length += t

        # reward logging
        log_f.write("{},{}\n".format(epochs, total_rewards_episode))
        log_f.flush()

        # stop training if avg_reward > solved_reward
        if running_reward / log_interval > solved_reward:
            print("###### Solved! ######")
            torch.save(ppo.policy.state_dict(), './preTrained/PPO_{}.pth'.format(env_name))
            break

        # visualize reward and episode length
        if epochs % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int(running_reward / log_interval)
            print("Episode {} \t Average length: {} \t Average reward: {}".format(epochs, avg_length, running_reward))
            avg_length = 0
            running_reward = 0

if __name__ == "__main__":
    main()