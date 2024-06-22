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
     
    cost = {
        'storm_paper':{
            'no_task':{ # ArmBase
                'null_disp': {
                    'total': None,
                    'weights':[],
                    'terms': [],
                    'terms_meaning': []
                    },
                'manipulability':{
                    'total': None,
                    'weights':[],
                    'terms': [],
                    'terms_meaning': []
                    },
                'horizon': {
                    'stop': {
                        'total': None,
                        'weights':[],
                        'terms': [],
                        'terms_meaning': []
                    },
                    'stop_acc': { # todo - must distinguish between this in stop_cost
                        'total': None,
                            'terms': None
                        },
                    'smooth': {
                        'total': None,
                        'terms': None
                        }
                    },
                'bound': {'total': None, 'terms': None},
                'ee_velocity': {'total': None, 'terms': None},
                'collision': {'self': {'total': None, 'terms': None}, 'primitive': {'total': None, 'terms': None}, 'voxel':  {'total': None, 'terms': None}}
            },
            'task':{ # ArmReacher
                'pose': {
                    'total': None,
                    'weights':[],
                    'terms': [],
                    'terms_meaning': ['error in orientation (of end effector) comparing to its orientation at goal state',
                                      'error in position (of end effector) comparing to its position at goal state']
                    }                        
                },
        },
        'rlpt': {'todo': 'todo'}
    } 

    
    # @staticmethod
    # def convergence_all():
    #     convergence_terms = ['orientation', 'position']
    #     return all([RealWorldState.goal_ee_convergence[c]['status'] for c in convergence_terms]) 
    
    # @staticmethod
    # def is_goal_posed_reached():
        
         
    # @staticmethod
    # def update_task_completion_status():
    #     """
    #     Setting true in the convergence flags if end effector reached goal location / goal orientation, based on approximation to 0 cost"""
        
    #     convergence_terms = ['orientation', 'position']
        
    #     # storm paper convergence test 
    #     for t in convergence_terms:
            
    #         RealWorldState.goal_ee_convergence[t]['status'] = RealWorldState.cost['storm_paper']['task']['pose'][t]['cost'] < 0.01
            
    #         # set the first itertaion of convergence - the reaching time
    #         if RealWorldState.goal_ee_convergence[t]['status']:
    #             is_first_convergence_iter = RealWorldState.goal_ee_convergence[t]['iter_of_convergence'] < 0
    #             if is_first_convergence_iter:
    #                 RealWorldState.goal_ee_convergence[t]['iter_of_convergence'] = RealWorldState.real_world_time 

