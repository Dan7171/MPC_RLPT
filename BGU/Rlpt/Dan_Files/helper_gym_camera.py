import gym
import isaacgym
from isaacgym import gymapi

# Initialize Isaac Gym
gym = gymapi.acquire_gym()

# Create a simulation environment
sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create a camera sensor
camera_props = gymapi.CameraProperties()
camera_props.width = 640
camera_props.height = 480
camera_props.enable_tensors = True

# Set the camera position and orientation
camera_pose = gymapi.Transform()
camera_pose.p = gymapi.Vec3(0, 1, 2)
camera_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

# Add the camera to the environment
env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, -1.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)
camera_handle = gym.create_camera_sensor(env, camera_props)

# Set the camera's pose
gym.set_camera_transform(camera_handle, env, camera_pose)

# Step the simulation and capture an image
gym.simulate(sim)
gym.fetch_results(sim, True)

# Get the camera image tensor
image_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_COLOR)

# Convert the tensor to a numpy array (if needed)
image_np = gym.get_tensor_gpu_data(image_tensor)
