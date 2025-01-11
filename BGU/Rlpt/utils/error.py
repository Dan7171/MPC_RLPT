from isaacgym import gymapi
import numpy as np



# Functions for converting gym objects to numpy vectors
def pose_as_ndarray(pose:gymapi.Transform) -> np.ndarray: # type: ignore
        """Converting a pose from a Transform object to a np.array in length 7 (indices 0-2 = position, 3-6 = rotation) """
        # get pos and rot as ndarray
        pos_np = pos_as_ndarray(pose.p)
        rot_np = rot_as_ndarray(pose.r)
        # concatenate to one vector in length 7
        return np.concatenate([pos_np, rot_np]) 
    
def pos_as_ndarray(pos:gymapi.Vec3) -> np.ndarray: # type: ignore
    
    """
    cast pos from gymapi.Vec3 to an ndarray in length 3 (np array - vector)
    """
    
    return np.array([pos.x, pos.y, pos.z])   
def rot_as_ndarray(rot:gymapi.Quat) -> np.ndarray: # type: ignore
    
    """
    cast rot from gymapi.Quat to an ndarray in length 4 (np array - vector)
    """
    
    return np.array([rot.x, rot.y, rot.z, rot.w])

# Error measurment functions:
def pose_error(curr_pose:gymapi.Transform, goal_pose:gymapi.Transform)-> np.float32: # type: ignore
    """
    return l2 norm between current and desired poses (each pose is both positon and rotation)  
    """
    return np.linalg.norm(pose_as_ndarray(curr_pose) - pose_as_ndarray(goal_pose))  
def pos_error(curr_pos:gymapi.Vec3, goal_pos:gymapi.Vec3)-> np.float32: # type: ignore
    """
    return l2 norm between current and desired positions (position is only the spacial location in environment ("x,y,z" coordinates))
    """
    return np.linalg.norm(pos_as_ndarray(curr_pos) - pos_as_ndarray(goal_pos))     
def rot_error(curr_rot:gymapi.Quat, goal_rot:gymapi.Quat)-> np.float32: # type: ignore
    """
    return l2 norm between current and desired rotations (each rotation is the quaternion - a 4 length vector)  
    """
    
    return np.linalg.norm(rot_as_ndarray(curr_rot) - rot_as_ndarray(goal_rot)) 