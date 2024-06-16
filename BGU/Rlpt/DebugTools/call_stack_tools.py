import inspect

def is_function_in_call_stack(func_name):
    # Get the current call stack
    stack = inspect.stack()
    # Iterate over the stack frames
    for frame_record in stack:
        # Check if the function name matches the desired function
        if frame_record.function == func_name:
            return True
    return False