from .call_stack_tools import is_function_in_call_stack

def is_real_world()-> bool:
    """
    return True <=> this the caller was called not as part of the call stack of rollout function (meaning its real) 
    """
    in_context_of_rollouts =  is_function_in_call_stack('rollout_fn')
    return not in_context_of_rollouts


class RealWorldState:
    world_id = 0 # may be useful when we use more than one world to train
    real_world_time = 0 # num of iterations passed in simulator, num of planning steps...
    goal_ee_convergence = {'position': {'status': False, 'iter': -1}, 'orientation': {'status': False, 'iter': -1}} # if goal state of end effector was reached and at which iteration it happend, for both orientation and position 

    cost = {
        'storm_paper':{
            'no_task':{
                'null_disp': {'cost': 100000, 'weight': 100000},
                'manipulability': {'cost': 100000, 'weight': 100000},
                'horizon':{'stop': {'cost': 100000, 'weight': 100000}, 'stop_acc': {'cost': 100000, 'weight': 100000}, 'smooth':  {'cost': 100000, 'weight': 100000}},
                'bound': {'cost': 100000, 'weight': 100000},
                'ee_velocity': {'cost': 100000, 'weight': 100000},
                'collision': {'self': {'cost': 100000, 'weight': 100000}, 'primitive': {'cost': 100000, 'weight': 100000}, 'voxel':  {'cost': 100000, 'weight': 100000}}
            },
            'task':{
                'pose': {'orientation': {'cost': 100000, 'weight': 15}, 'position': {'cost': 100000, 'weight': 1000}},                
            },
        },
        'rlpt': {'todo': 'todo'}
    } 


    @staticmethod
    def convergence_all():
        convergence_terms = ['orientation', 'position']
        return all([RealWorldState.goal_ee_convergence[c]['status'] for c in convergence_terms]) 
    
    @staticmethod
    def update_convergece_status():
         """
         Setting true in the convergence flags if end effector reached goal location / goal orientation, based on approximation to 0 cost"""
         convergence_terms = ['orientation', 'position']
         for c in convergence_terms:
            converged_c = RealWorldState.cost['storm_paper']['task']['pose'][c]['cost'] < 0.01# cost of convergence term is low enough
            RealWorldState.goal_ee_convergence[c]['status'] = True if converged_c else False # set to converged!
            if converged_c and RealWorldState.goal_ee_convergence[c]['iter'] < 0:    
                RealWorldState.goal_ee_convergence[c]['iter'] = RealWorldState.real_world_time # now it converged
