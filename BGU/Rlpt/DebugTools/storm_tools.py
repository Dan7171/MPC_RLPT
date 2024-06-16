from .call_stack_tools import is_function_in_call_stack

def is_real_world()-> bool:
    """
    return True <=> this the caller was called not as part of the call stack of rollout function (meaning its real) 
    """
    in_context_of_rollouts =  is_function_in_call_stack('rollout_fn')
    return not in_context_of_rollouts