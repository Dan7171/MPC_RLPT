import os
import random
import gymnasium as gym
import numpy as np
import torch
from Rlpt.drl.rainbow_rlpt.dqn_agent import DQNAgent
import base64
import glob
import io
import os
from IPython.display import HTML, display


env = gym.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")

# env = gym.make("CartPole-v1", max_episode_steps=200, render_mode="human")
seed = 777

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
random.seed(seed)
seed_torch(seed)

# parameters
num_frames = 1000 # 10000
memory_size = 10000
batch_size = 128
target_update = 100

# train
agent = DQNAgent(env, memory_size, batch_size, target_update, seed)
agent.train(num_frames)
video_folder="videos/rainbow"
print("test started")
agent.test(video_folder=video_folder)




def ipython_show_video(path: str) -> None:
    """Show a video at `path` within IPython Notebook."""
    if not os.path.isfile(path):
        raise NameError("Cannot access: {}".format(path))

    video = io.open(path, "r+b").read()
    encoded = base64.b64encode(video)

    display(HTML(
        data="""
        <video width="320" height="240" alt="test" controls>
        <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
        </video>
        """.format(encoded.decode("ascii"))
    ))


def show_latest_video(video_folder: str) -> str:
    """Show the most recently recorded video from video folder."""
    list_of_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    latest_file = max(list_of_files, key=os.path.getctime)
    ipython_show_video(latest_file)
    return latest_file


# latest_file = show_latest_video(video_folder=video_folder)
# print("Played:", latest_file)