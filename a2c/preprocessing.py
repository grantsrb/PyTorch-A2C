from skimage.color import rgb2grey
import numpy as np

def normalize_prep(pic):
    new_pic = 3*(pic - 255/2)/(255/2)
    return new_pic[None]

def null_prep(pic):
    return pic[None]

def pong_prep(pic):
    pic = pic[35:195] # crop
    pic = pic[::2,::2,0] # downsample by factor of 2
    pic[pic == 144] = 0 # erase background (background type 1)
    pic[pic == 109] = 0 # erase background (background type 2)
    pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
    return pic[None]

def breakout_prep(pic):
    pic = pic[35:195,8:-8] # crop
    pic = pic[::2,::2,0] # downsample by factor of 2
    pic = rgb2grey(pic)
    return pic[None]

def snake_prep(pic):
    new_pic = np.zeros(pic.shape[:2],dtype=np.float32)
    new_pic[:,:][pic[:,:,0]==1] = 1
    new_pic[:,:][pic[:,:,0]==255] = 1.5
    new_pic[:,:][pic[:,:,1]==255] = 0
    new_pic[:,:][pic[:,:,2]==255] = .33
    pic = new_pic
    return new_pic[None]
