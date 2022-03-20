import torch
from PIL import Image
import gym
from collections import deque

# Action and reward issue

class Phi_fn_cartpole:

    def __init__(self, stack_length):

        self.stack_length = stack_length
        self.transform = lambda x: np.asarray(x.convert('YCbCr').getchannel('Y').resize((84, 84), resample=PIL.IMAGE.BILINEAR))
        
        self.framestack = deque([], maxlen=self.stack_length)
    
    def get_initial_framestack(self, env):

        def reset_variables():
            
            nonlocal sl_count = 0 # Stacklength counter
            nonlocal sf_count = 0 # Skip length counter
            nonlocal o = env.reset()
            nonlocal a = env.action_space.sample()
            nonlocal done = False
            nonlocal screen_buffer = deque([], maxlen=2)

        reset_variables()
        
        while sl_count < self.stack_length:

            for sf_count in skip_frames:
                o, r, done, _ = env.step(a)
                screen_buffer.update(env.render(mode='rgb_array'))

                if done and sl_count != self.stack_length:
                    self.framestack = deque([], maxlen=self.stack_length)
                    reset_variables()

            if not (done and sl_count != self.stack_length):
                self.update_framestack(screen_buffer)
                sl_count += 1
                a = env.action_space.sample()
        
        return torch.stack(self.framestack, dim=0), screen_buffer

    def update_framestack(self, screen_buffer):

        # Update framestack with latest screen buffer after skipping frames
        obs = np.maximum(screen_buffer[-1], screen_buffer[-2])
        obs = np.divide(obs, 255, dtype=float)
        obs = Image.fromarray(obs.transpose((2, 0, 1)))
        self.framestack.append(torch.as_tensor(self.transform(obs)))