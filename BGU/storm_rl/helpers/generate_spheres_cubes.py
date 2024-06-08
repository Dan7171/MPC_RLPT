import itertools

def generate_sphere_data():
    sphere_data = {}
    radius_values = [round(0.02 + i * 0.009, 2) for i in range(10)]
    for i, radius in enumerate(radius_values, start=1):
        sphere_data[f"sphere{i}"] = {'radius': radius, 'position': [0.0, 0.0, 0.0]}
    return sphere_data

def generate_cube_data():
    cube_data = {}
    dimensions = [0.1, 0.2, 0.3]
    for i, dims in enumerate(itertools.product(dimensions, repeat=3), start=1):
        cube_data[f"cube{i}"] = {'dims': list(dims), 'pose': [0.0, 0.0, 0.0, 0, 0, 0, 1.0]}
    return cube_data

def print_object_info():
    print("sphere:")
    for sphere_name, sphere_info in generate_sphere_data().items():
        print(f"  {sphere_name}:")
        for key, value in sphere_info.items():
            if key == 'radius':
                print(f"    {key}: {value} # meters")
            elif key == 'position':
                print(f"    {key}: {value}")
    
    print("\ncube:")
    for cube_name, cube_info in generate_cube_data().items():
        print(f"  {cube_name}:")
        for key, value in cube_info.items():
            if key == 'dims':
                print(f"    {key}: {value}")
            elif key == 'pose':
                print(f"    {key}: {value}")

if __name__ == "__main__":
    print_object_info()
