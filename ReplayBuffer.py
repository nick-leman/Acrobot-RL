from collections import deque

class ReplayBuffer(object):
      def __init__(self, size):
        self.buffer = deque(maxlen=size)

      def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

      def __len__(self):
        return len(self.buffer)

      def sample(self, num_samples):
        indexes = np.random.choice(len(self.buffer), num_samples)
        states, actions, rewards, next_states, dones=np.array(self.buffer)[indexes,:].T
        states=[x[0] if len(x)==2 else x for x in states] # debeque return empty set with actual value should be fixed 

        return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(dones)