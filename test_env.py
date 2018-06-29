import gym
import sys
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2grey

def preprocess(pic, env_type='snake-v0'):
    if env_type == "Pong-v0":
        pic = pic[35:195] # crop
        pic = pic[::2,::2,0] # downsample by factor of 2
        pic[pic == 144] = 0 # erase background (background type 1)
        pic[pic == 109] = 0 # erase background (background type 2)
        pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
    elif 'Breakout' in env_type:
        pic = pic[35:195] # crop
        pic = rgb2grey(pic)
        pic = pic[::2,::2] # downsample by factor of 2
        pic = pic/255 # everything else (paddles, ball) just set to 1
    elif env_type == "snake-v0":
        new_pic = np.zeros(pic.shape[:2],dtype=np.float32)
        new_pic[:,:][pic[:,:,0]==1] = 1
        new_pic[:,:][pic[:,:,0]==255] = 1.5
        new_pic[:,:][pic[:,:,1]==255] = 0
        new_pic[:,:][pic[:,:,2]==255] = .33
        pic = new_pic
    return pic[None]

# gym_type = str(input("Gym Type:"))
gym_type = 'BreakoutNoFrameskip-v4'

env = gym.make(gym_type)

obs = env.reset()
obs = preprocess(obs, env_type=gym_type)
viewer = plt.imshow(obs.squeeze())
plt.draw()
plt.pause(.1)

action = 1

prev_obs = 0
while action != -1:
    obs, rew, done, _ = env.step(action)
    #env.render()
    prepped = preprocess(obs, env_type=gym_type).squeeze()
    obs = prepped-.5*prev_obs
    prev_obs = prepped
    viewer.set_data(obs)
    plt.draw()
    print("reward:",rew)
    if done:
        env.reset()
    inp = str(input())
    if inp == 'a':
        action = 3
    elif inp == 'd':
        action = 2
    elif inp == 'w':
        action = 1
    elif inp == 'q':
        action = -1
