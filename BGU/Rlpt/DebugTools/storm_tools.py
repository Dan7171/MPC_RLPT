from .call_stack_tools import is_function_in_call_stack
import copy
import torch
def is_real_world()-> bool:
    """
    return True <=> this the caller was called not as part of the call stack of rollout function (meaning its real) 
    """
    in_context_of_rollouts =  is_function_in_call_stack('rollout_fn')
    return not in_context_of_rollouts

def tensor_to_float(torch_tensor: torch.tensor):
    """ getting the first value from the tensor
    Args:
        tensor (_type_): _description_

    Returns:
        _type_: _description_
    """
    return float(torch_tensor)

class RealWorldState:
    
    @staticmethod
    def reset_cost():
        return copy.deepcopy(RealWorldState.cost_template)
    
    @staticmethod
    def reset(time):
        RealWorldState.real_world_time = time
        RealWorldState.cost = RealWorldState.reset_cost()
        pass
    
    
    # world_id = 0 # may be useful when we use more than one world to train
    real_world_time = 0 # num of iterations passed in simulator, num of planning steps...
    cur_step_cost = 0
    total_cost = 0 
    cost_template = {
        'storm_paper':{
            'ArmBase':{ # ArmBase
                'null_disp': {  # v
                    'total': None,
                    'weights':[],
                    'terms': [],
                    'terms_meaning': []
                    },
                'manipulability':{  # v
                    'total': None,
                    'weights':[],
                    'terms': [],
                    'terms_meaning': []
                    },
                'horizon': { 
                    'stop': {  # v
                        'total': None,
                        'weights':[],
                        'terms': [],
                        'terms_meaning': []
                        },
                    'stop_acc': {  # v
                        'total': None,
                        'weights':[],
                        'terms': [],
                        'terms_meaning': []
                        },
                    'smooth': {  # v
                        'total': None,
                        'weights':[],
                        'terms': [],
                        'terms_meaning': []
                        }
                    },
                'bound': {  
                        'total': None,
                        'weights':[],
                        'terms': [],
                        'terms_meaning': []
                        },
                'ee_vel': { # end effector velocity?  
                        'total': None,
                        'weights':[],
                        'terms': [],
                        'terms_meaning': []
                        },
                'collision': { # optional - if not no_coll
                    'self': {  
                        'total': None,
                        'weights':[],
                        'terms': [],
                        'terms_meaning': []
                        }, 
                    'primitive': {  #v
                        'total': None,
                        'weights':[],
                        'terms': [],
                        'terms_meaning': []
                        }, 
                    'voxel':  {  #v
                        'total': None,
                        'weights':[],
                        'terms': [],
                        'terms_meaning': []
                        }
                    },
            
            },
            
            'ArmReacher':{ # ArmReacher - goal state related costs
                'goal': { # also called pose_cost (in paper too probably)
                    'total': None,
                    'weights':[],
                    'terms': [],
                    'terms_meaning': ['orientation of end effector ', 'position of end effector ']
                    },
                
                'joint_l2': { 
                    'total': None,
                    'weights':[],
                    'terms': [],
                    'terms_meaning': []
                    },
                
                'zero_acc': { # zero acceleration?(optional) 
                    'total': None,
                    'weights':[],
                    'terms': [],
                    'terms_meaning': []
                    },
                'zero_vel': { # zero velocity? (optional) 
                    'total': None,
                    'weights':[],
                    'terms': [],
                    'terms_meaning': []
                    }                                                
                },
            
            
        },
        'rlpt': {'todo': 'todo'}
    }
    
    @staticmethod
    def clean_view(costs:dict):
        costs_copy = copy.deepcopy(costs)
        def modifier(d):
            keys_to_pop = []    
            for k,v in d.items():
                if type(v) == dict:
                    modifier(v)
                elif v in [None, []]:
                    keys_to_pop.append(k)    
            for k in keys_to_pop:
                d.pop(k)
        
        modifier(costs_copy)
        costs_copy = RealWorldState.replace_1item_tensors_in_floats(costs_copy)
        new_dict = {}
        for k,v in costs_copy['storm_paper']['ArmBase'].items():
            new_dict[k] = v
        for k,v in costs_copy['storm_paper']['ArmReacher'].items():
            new_dict[k] = v
        return new_dict
        
    @staticmethod
    def replace_1item_tensors_in_floats(costs:dict):
        costs_copy = copy.deepcopy(costs)
        
        def modifier(d):    
            for k,v in d.items():
                if type(v) == dict:
                    modifier(v)
                elif type(v) == list:
                    for i,list_item in enumerate(v):
                        if type(list_item) == torch.Tensor:
                            v[i] = tensor_to_float(list_item) # replace tensor with its value
                
                elif type(v) == torch.Tensor:
                    d[k] = tensor_to_float(v) # replace tensor with its value
        
        modifier(costs_copy)
        return costs_copy
                            
                            
            
    @staticmethod
    def get_costs_weight_list(costs:dict, key_chain=[]):
        ans = []
        for k,v in costs.items():
            key_chain.append(k)
            if k == 'weights':
                ans.append((copy.copy(key_chain)[:-1], v))
            elif type(v) == dict:
                ans += RealWorldState.get_costs_weight_list(v, key_chain)
            key_chain.pop()
        return ans
                
                
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

if __name__ == '__main__':
    ans = RealWorldState.get_costs_weight_list(RealWorldState.cost_template)
    for item in ans:
        print(item)