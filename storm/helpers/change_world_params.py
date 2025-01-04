import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path

world_file = 'collision_primitives_3d_origina.yml'

world_yml = join_path(get_gym_configs_path(), world_file)

def select_random_indexes(first, last, n):
    print("ELIAS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Define the range of indexes from first to last (inclusive)
    indexes = list(range(first, last))
    
    # Randomly determine how many indexes to select (between 0 and n)
    num_indexes_to_select = random.randint(0, n)
    
    # Randomly select the determined number of indexes from the list
    selected_indexes = random.sample(indexes, min(num_indexes_to_select, len(indexes)))
    
    return selected_indexes

def generate_random_position(n):
    # Generate random position in a grid divided into n^2 blocks
    x = random.uniform(-2/n, 2/n)
    y = random.uniform(-2/n, 2/n)
    z = random.uniform(0.1, 0.7)
    return [x, y, z]

def generate_random_quaternion():
    # Generate random quaternion representing a rotation
    euler_angles = [random.uniform(0, 2*math.pi) for _ in range(3)]
    rotation = R.from_euler('xyz', euler_angles)
    quaternion = rotation.as_quat()
    return quaternion.tolist()

def open_yaml(world_yml):
    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)
    # print(f"world_params: {world_params}")
    return world_params

def get_objects_by_indexes(world_params, indexes):
    coll_objs = world_params['world_model']['coll_objs']
    
    # Flatten the dictionary into a list of (key, value) pairs
    objects = []
    for obj_type, obj_dict in coll_objs.items():
        for obj_name, obj_info in obj_dict.items():
            objects.append((obj_name, obj_info))
    
    # Get the objects corresponding to the provided indexes
    selected_objects = []
    for index in indexes:
        if 0 <= index < len(objects):
            selected_objects.append(objects[index])
        else:
            raise IndexError(f"Index {index} out of range")
    
    return selected_objects

def get_base_name(name):
    base_name = ''.join([char for char in name if char.isalpha()])
    return base_name

def randomize_pos(obj, base_name):

    position = generate_random_position(2)
    if base_name == 'cube':
        quat = generate_random_quaternion()
        return position + quat
    else:
        return position
    
def modify_dict(world_yml):
    # Open dictionary
    world_params = open_yaml(world_yml)
    # Select random indexes
    indexes_spheres = select_random_indexes(0, 10, 5)
    indexes_cubes = select_random_indexes(11, 35, 15)
    indexes = indexes_spheres + indexes_cubes
    print(f"indexes: {indexes}")
    # Get objects from dictionary
    selected_objects = get_objects_by_indexes(world_params, indexes)
    print(f"selected_objects: {selected_objects}")

    # Create new dictionary
    compressed_world_params = {'world_model': {'coll_objs': {'sphere': {},'cube': {}}}}

    sphere_index = 1
    cube_index = 1
    for i in range(len(indexes)):
        obj = selected_objects[i]
        name = obj[0]
        base_name = get_base_name(name)

        new_pos = randomize_pos(obj, base_name)

        if base_name == 'sphere':
            # Modify dict
            world_params['world_model']['coll_objs'][base_name][name]['position'] = new_pos
            # Add to compressed dict
            radius_position = {}
            radius_position['radius'] = world_params['world_model']['coll_objs'][base_name][name]['radius']
            radius_position['position'] = world_params['world_model']['coll_objs'][base_name][name]['position']
            compressed_world_params['world_model']['coll_objs'][base_name][base_name + str(sphere_index)] = radius_position
            sphere_index += 1
        elif base_name == 'cube':
            print("Cube added !!!")
            # Modify dict
            world_params['world_model']['coll_objs'][base_name][name]['pose'] = new_pos
            # Add to compressed dict
            dims_pose = {}
            dims_pose['dims'] = world_params['world_model']['coll_objs'][base_name][name]['dims']
            dims_pose['pose'] = world_params['world_model']['coll_objs'][base_name][name]['pose']
            compressed_world_params['world_model']['coll_objs'][base_name][base_name + str(cube_index)] = dims_pose
            cube_index += 1

        dims_pose = {}
        dims_pose['dims'] = world_params['world_model']['coll_objs']['cube']['cube28']['dims']
        dims_pose['pose'] = world_params['world_model']['coll_objs']['cube']['cube28']['pose']
        compressed_world_params['world_model']['coll_objs']['cube']['cube28'] = dims_pose

    print(f"modyfied dict: {world_params}")
    print(f"new dict: {compressed_world_params}")

    return world_params, indexes, compressed_world_params




modify_dict(world_yml)