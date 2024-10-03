from isaacgym import gymapi
import numpy as np
from scipy.spatial.transform import Rotation as R  # Use this to handle quaternions

# Function to generate a point cloud for a sphere
def generate_sphere_point_cloud(radius, num_points=1000):
    """
    Generate a point cloud of a sphere.
    Args:
    - radius: radius of the sphere
    - num_points: number of points to generate
    Returns:
    - points: Nx3 array of points in the sphere's point cloud
    """
    points = []
    
    # Uniformly distribute points on the surface of the sphere using spherical coordinates
    for i in range(num_points):
        theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
        phi = np.random.uniform(0, np.pi)  # Polar angle

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        points.append([x, y, z])

    return np.array(points)




# Function to generate a point cloud from robot link locations
def generate_robot_point_cloud(gym, env):
    FRANKA_ACTOR_IDX = 0
    franka_handle = gym.get_actor_handle(env, FRANKA_ACTOR_IDX)
    
    num_dof = gym.get_actor_dof_count(env, franka_handle)  # Number of degrees of freedom

    point_cloud = []

    # Loop through each joint or link of the robot (could also be objects in the scene)
    for i in range(num_dof):
        link_state = gym.get_actor_rigid_body_states(env, franka_handle, gymapi.STATE_POS)[i]
        pos = link_state['pose']['p']  # This is a gymapi.Vec3 (position)
        point_cloud.append([pos.x, pos.y, pos.z])

    return np.array(point_cloud)  # Convert list of points to NumPy array (3D point cloud)


# Function to generate a point cloud for a box and rotate it
def generate_rotated_box_point_cloud(box_size, rotation_quat, resolution=0.1):
    """
    Generate a rotated point cloud of a box surface.
    Args:
    - box_size: (x, y, z) dimensions of the box
    - rotation_quat: Quaternion (w, x, y, z) representing the rotation of the box
    - resolution: distance between adjacent points
    Returns:
    - rotated_points: Nx3 array of rotated points in the box's point cloud
    """
    # Generate the point cloud in local (unrotated) box coordinates
    x_range = np.arange(-box_size[0] / 2, box_size[0] / 2, resolution)
    y_range = np.arange(-box_size[1] / 2, box_size[1] / 2, resolution)
    z_range = np.arange(-box_size[2] / 2, box_size[2] / 2, resolution)

    points = []

    # Front and back faces (x, y plane)
    for x in x_range:
        for y in y_range:
            points.append([x, y, box_size[2] / 2])  # Front face
            points.append([x, y, -box_size[2] / 2])  # Back face

    # Left and right faces (y, z plane)
    for y in y_range:
        for z in z_range:
            points.append([box_size[0] / 2, y, z])  # Right face
            points.append([-box_size[0] / 2, y, z])  # Left face

    # Top and bottom faces (x, z plane)
    for x in x_range:
        for z in z_range:
            points.append([x, box_size[1] / 2, z])  # Top face
            points.append([x, -box_size[1] / 2, z])  # Bottom face

    # Convert the points to a NumPy array
    points = np.array(points)

    # Apply rotation to the points
    r = R.from_quat(rotation_quat)  # Create a rotation object from the quaternion
    rotated_points = r.apply(points)  # Apply rotation to all points

    return rotated_points

if __name__ == '__main__':
    
    # Example usage
    box_size = [2.0, 1.0, 0.5]  # Box dimensions (2m x 1m x 0.5m)
    resolution = 0.1  # 10 cm resolution
    rotation_quat = [0.92388, 0.0, 0.38268, 0.0]  # Example quaternion (w, x, y, z) for 45-degree rotation around Z-axis
    rotated_box_point_cloud = generate_rotated_box_point_cloud(box_size, rotation_quat, resolution)
    print("Rotated box point cloud:", rotated_box_point_cloud)

    
    # Example usage
    radius = 1.0  # Sphere radius
    num_points = 1000  # Number of points to generate
    sphere_point_cloud = generate_sphere_point_cloud(radius, num_points)

    print("Sphere point cloud:", sphere_point_cloud)

    
    