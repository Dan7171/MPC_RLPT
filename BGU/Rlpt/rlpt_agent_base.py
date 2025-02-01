from abc import abstractmethod
from collections import deque
import copy
import csv
from importlib import metadata
from os import access
import os
import time
from turtle import pos, position
from typing import Any, Dict, List, Set, Tuple, Union
from click import BadArgumentUsage, BadParameter
from networkx import union
from storm_kit.mpc import task
from sympy import epath
import torch
from yaml import compose
import numpy as np
import pandas as pd
from BGU.Rlpt import rlpt_agent
from BGU.Rlpt.DebugTools.globs import GLobalVars
from BGU.Rlpt.utils.error import pos_error, pose_as_ndarray, rot_error
from BGU.Rlpt.utils.type_operations import torch_tensor_to_ndarray
from BGU.Rlpt.utils.utils import color_print, goal_test, is_sorted_decreasing

META_DATA_SIGNATURE_ETL = 'MD'
# AT_SIGNATURE_ETL = 'AT'
ST_SIGNATURE_ETL = 'ST'
def get_obj_type(obj_properties):
    if 'radius' in obj_properties:
        return 'sphere'
    elif 'dims' in obj_properties:
        return 'cube'

def _merge_dict(original, update):
    new = copy.deepcopy(original)
    new.update(update)
    return new

def increasing_linear_array(k):
    # Create linearly increasing values from 0 to 1
    assert k >= 1 and k == int(k)
    values = np.linspace(1/k, 1, k)    
    if k == 1:
        return np.array([1])
    # Normalize to sum to 1
    return values / np.sum(values)

class rlptAgentBase:       
    def __init__(self,base_pos_gym: np.ndarray, participating_storm:dict, not_participating_storm:dict,col_obj_handles:dict, action_space:list, etl_logging:bool,reward_cfg:dict, training_mode=True):
        """
        Summary:
            initializng a reinforcement learning parameter tuning agent.
            
        Args:
            participating_storm (dict): Participating objects initial states {name:attributes}. Positions and orientations are in storm coordinate system.
            not_participating_storm (dict): Not participating objects initial states {name: attributes}. Positions and orientations are in storm coordinate system.
            col_obj_handles dict: a dict which keys are the names of collision objects (participating or not) and values are the object's handle (int)
            action_space (list): a collection of all possible actions which can be selected (all combinations of hyper parameters that can be selected at a single stime step)
            training_mode: (bool): if training or in test mode
        """
        
        # self.completed_optimization_steps_cntr = 0
        self.training_mode = training_mode # new
        self.trainig_episodes_done = 0 # new
        self.trainig_steps_done = 0 # new
        self.test_episodes_done = 0 # new
        self.test_steps_done = 0 # new
        self.is_milestone_cnt_changed = False # New
        self.reward_cfg = reward_cfg # new
        self.pose_err_milestones_reward_cfg = self.reward_cfg['pose_reward']['pose_err_milestones'] # new
        self.etl_logging = etl_logging
        self.action_space:list = action_space
        self.col_obj_handles:dict = col_obj_handles
        self.all_coll_obj_names_sorted:list = sorted(list(self.col_obj_handles.keys()), key=lambda x: self.col_obj_handles[x]) # sorted by obj handle
        all_coll_objs_s0 = _merge_dict(participating_storm, not_participating_storm)
        
        if len(all_coll_objs_s0) != len(participating_storm) + len(not_participating_storm):
            raise BadArgumentUsage("participating and non participating objects must contain objects with unique names") 
        
        # self.max_col_objs = max_col_objs        
        self.col_obj_s0_flat_states:dict[str,np.ndarray]= self.flatten_coll_obj_states(all_coll_objs_s0) # {'obj name': flattened objected state in storm cs([5,1,3,0.4...])}
        col_obj_s0_sorted_concat:np.ndarray = self.flatten_sorted_coll_objs_states(self.col_obj_s0_flat_states) # concatenated flattened objected states sorted by obj handle
        self.max_horizon = self._get_max_h() # maximun horizon in action space
        
        # define the state representation configuration
        rlpt_cfg = GLobalVars.rlpt_cfg
        assert rlpt_cfg is not None
        self.seed = rlpt_cfg['seed']
        state_represantation_config = rlpt_cfg['agent']['model']['state_representation']
         
        # all possible state representation components and their dimensions
        if state_represantation_config['pi_mppi_means'] == 'max':
            self.pi_mppi_means_horizon_in_st = self.max_horizon 
        elif not state_represantation_config['pi_mppi_means']:
            self.pi_mppi_means_horizon_in_st = 0
        else:
            assert state_represantation_config['pi_mppi_means'] == int(state_represantation_config['pi_mppi_means'])
            self.pi_mppi_means_horizon_in_st = state_represantation_config['pi_mppi_means']
        
        state_var_to_dim =  {
            'robot_dofs_positions': 7, # 1 scalar (angular position w.r to origin (0)) for each dof (joint) of the 7 dofs ,
            'robot_dofs_velocities': 7, # similarly to positions, an angular velocity on each dof 
            'goal_pose':  7, # position (3), orientation (4)
            'coll_objs': 7 * len(col_obj_s0_sorted_concat), # obj size (7) times num of collision objs
            'robot_base_pos': 3, # xyz (position only)
            'prev_action_idx': 1, # the index of the previous action which was taken
            'pi_mppi_means':  7 * self.pi_mppi_means_horizon_in_st, # MPPI policy (H gaussians) means: 7 distribution means (one for each dof) for max- H actions
            'pi_mppi_covs': 7 ,# MPPI policy (H gaussians) covariances: 7 covariances of those means (unlike the means, the covs remain the same for the whole horizon)
            'ee_err_milestones': 1,
            't': 1 # current time step 
        }
        
        
        if self.pose_err_milestones_reward_cfg['use']:
            pos_errs_of_milestones = [milestone[0] for milestone in self.pose_err_milestones_reward_cfg['milestones']]
            rot_errs_of_milestones = [milestone[1] for milestone in self.pose_err_milestones_reward_cfg['milestones']]
            print(rot_errs_of_milestones, rot_errs_of_milestones)
            assert is_sorted_decreasing(pos_errs_of_milestones)
            assert is_sorted_decreasing(rot_errs_of_milestones)
            state_represantation_config['ee_err_milestones'] = True
        

        # add state components names. That's what determines the ordere of components int the long state which represented as 1d vecotr   
        self.st_componentes_ordered = []
        for var in state_represantation_config.keys(): 
            if state_represantation_config[var]:
                self.st_componentes_ordered.append(var)
                
        if self.reward_cfg['pose_reward']['ep_tail_err_reward']['use']:
            assert 't' in self.st_componentes_ordered, 'self.ep_tail_err_reward_weights is initialized only if t is in state representation. To ep_tail_err_reward, user must represent t in state (to prevent confusing ambiguity in reward function)'            
            self.ep_tail_err_reward_weights = increasing_linear_array(self.reward_cfg['pose_reward']['ep_tail_err_reward']['tail_len']) # increasing summed to 1 array of weights (non negative)

        # define the current state s(t) (initially s0)
        self.current_st = {component: np.array([]) for component in self.st_componentes_ordered} # component (name) to the component values
        if 'robot_base_pos' in self.st_componentes_ordered:
            self.current_st['robot_base_pos'] = base_pos_gym
        if 'coll_objs' in self.st_componentes_ordered:
            self.current_st['coll_objs'] = col_obj_s0_sorted_concat
        if 'prev_action_idx' in self.st_componentes_ordered:
            self.current_st['prev_action_idx'] = np.array([-1])
        if 'ee_err_milestones' in self.st_componentes_ordered:
            self.current_st['ee_err_milestones'] = np.array([0])
        
        
        self.st_componentes_ordered_dims = [state_var_to_dim[component] for component in self.st_componentes_ordered]         
        self.component_to_location_list:list = self.get_component_with_range_list()         
        self.component_to_location_dict:dict = {name_to_loc[0]:name_to_loc[1] for name_to_loc in self.component_to_location_list}
        
        
        self.st_dim:int = sum(self.st_componentes_ordered_dims) # len of each state s(t) (NN input length)
        self.at_dim: int = len(action_space) # new
        self.st_legend = self.get_states_legend() # readable shape of the state 
        self.shared_action_features, self.unique_action_features_by_idx = self.action_space_info() # mostly for printing
        self.tuning_enabled = rlpt_cfg['agent']['action_space']['tuning_enabled'] if 'tuning_enabled' in rlpt_cfg['agent']['action_space'] else True 
        self.etl_col_names = []
    
    def initialize_etl(self,etl_file_path:str, st_metadata_labels:list,at_metadata_labels:list,rt_metadata_labels:list, snext_metadata_lables:list,step_metadata_labels:list, optim_metadata_labels:list=[],special_labels:list=[]):
        """  
        Initializing the etl file with proper lables
        Args:
            etl_file_path: path to etl (csv file)
            st_metadata_labels : st meta data labels
            at_metadata_labels : at meta data labels
            rt_metadata_labels : rt meta data labels
            snext_metadata_lables:s(t+1) meta data labels
            optim_metadata_labels (list, optional): optimization meta data labels. Defaults to [].
            step_metadata_labels: step (from st to st+1) meta data labeles
            special_labels (list, optional): Any extra labels that user wants to add to etl. Defaults to [].
        """        
        
        st_metadata_labels = [f'st{META_DATA_SIGNATURE_ETL}{title}' for title in st_metadata_labels]
        at_metadata_labels = [f'at{META_DATA_SIGNATURE_ETL}{title}' for title in at_metadata_labels]
        rt_metadata_labels = [f'rt{META_DATA_SIGNATURE_ETL}{title}' for title in rt_metadata_labels]
        snext_metadata_labels = [f's(t+1){META_DATA_SIGNATURE_ETL}{title}' for title in snext_metadata_lables]
        optim_metadata_labels = [f'optim{META_DATA_SIGNATURE_ETL}{title}' for title in optim_metadata_labels]
        step_metadata_labels = [f'step{META_DATA_SIGNATURE_ETL}{title}' for title in step_metadata_labels]
        
        # st_labels = self.get_states_legend()
        # st_labels = ['st_' + pair[1] for pair in st_labels]
        st_labels = self.st_componentes_ordered
        st_labels = ['st_' + labelname for labelname in st_labels]
        at_labels_unique_features = []
        if len(self.unique_action_features_by_idx):
            at_labels_unique_features = self.unique_action_features_by_idx[0].keys()
        at_labels_unique_features = ['at_' + k for k in at_labels_unique_features]
        at_labels_shared_features = []
        if len(self.shared_action_features):
            at_labels_shared_features = self.shared_action_features.keys()
        at_labels_shared_features = ['at_' + k for k in at_labels_shared_features]
        col_names = ['t_total', 'ep_id', 't_ep', *st_labels, *st_metadata_labels, 'at_id',  *at_labels_unique_features, *at_labels_shared_features, *at_metadata_labels,'rt',*rt_metadata_labels,*snext_metadata_labels, *step_metadata_labels]
        if self.training_mode:
            col_names.extend(optim_metadata_labels)
        col_names.extend(special_labels)     
        with open(etl_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(col_names)
        print(f'new etl file was initialized: {etl_file_path}')
        return col_names
            
        
    
    # def update_etl(self, st, action_id,rt,ep_num,ts,at_meta_data,contact_detected,step_duration,ee_pos_error, ee_rot_error,etl_file_path, optim_meta_data, goal_reached):
    def update_etl(self,etl_file_path, episode_logging_info,special_labels=[]):
        
        def get_md_category_label_list(md_category:str):
            transition_meta_data_in_cat = episode_logging_info[f'{md_category}{META_DATA_SIGNATURE_ETL}'][0]
            if not len(transition_meta_data_in_cat): # no meta data
                return []
            return list(transition_meta_data_in_cat.keys())
        

        # print(f'debug 3: {len(episode_logging_info["t_ep"])}, {len(episode_logging_info["at_id"])}')
        
        if not os.path.exists(etl_file_path):
        
            md_categories = ['st', 'at', 'rt', 's(t+1)','step', 'optim']
            md_labels = []
            for md_cat in md_categories:
                md_labels.append(get_md_category_label_list(md_cat))
            self.etl_col_names = self.initialize_etl(etl_file_path, *md_labels, special_labels)
        
        else:
            with open(etl_file_path, mode='r') as file:
                reader = csv.reader(file)
                self.etl_col_names = next(reader)
            
        with open(etl_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            total_transition_num = len(episode_logging_info['t_ep']) # num of transitions in episode (num of rows to write)
            for transition_index in range(total_transition_num):
                log_row = []
                for i in range(len(self.etl_col_names)):
                    colname = self.etl_col_names[i]
                    if META_DATA_SIGNATURE_ETL in colname:
                        md_cat_with_signature = colname
                        md_cat_wo_signature, key_in_category = md_cat_with_signature.split(META_DATA_SIGNATURE_ETL)
                        logging_info_key = f'{md_cat_wo_signature}{META_DATA_SIGNATURE_ETL}'
                        log_row.append(episode_logging_info[logging_info_key][transition_index][key_in_category])
                        
                    elif colname.startswith('st_') and (i == 0 or not self.etl_col_names[i-1].startswith('st_')): # check if its the first signed val
                        st_parsed = list(self.parse_st(episode_logging_info['st'][transition_index]).values()) 
                        log_row.extend(st_parsed)
                        
                    elif colname == 'at_id': # check if its the first signed val
                        at_id = episode_logging_info[colname][transition_index]
                        at_unique_features = []
                        if len(self.unique_action_features_by_idx):
                            at_unique_features = list(self.unique_action_features_by_idx[at_id].values())
                        at_shared_features = []
                        if len(self.shared_action_features):
                            at_shared_features = list(self.shared_action_features.values())
                        log_row.append(at_id)
                        log_row.extend(at_unique_features)
                        log_row.extend(at_shared_features)
                    
                    elif colname.startswith('at_') and not colname.startswith('at_id') or colname.startswith('st_'):
                        continue
                    
                    elif colname == 'ep_id':
                        log_row.append(self.get_episodes_done()) # the number of done episodes is the index at this moment, because the counter is increased after this logging    
                    
                    else:
                        log_row.append(episode_logging_info[colname][transition_index])

                
                writer.writerow(log_row)
               
                    
                        
        
                     
    
    def flatten_coll_obj_states(self, all_coll_objs_st):
        
        out: dict[str,np.ndarray] = {}
        for obj_name in self.all_coll_obj_names_sorted:
            nested_obj_state = list(all_coll_objs_st[obj_name].values()) 
            flattened_obj_state = np.concatenate([np.atleast_1d(x) for x in nested_obj_state]) # [x,[y],z,[t,w]]] -> [[x],[y],[z],[t,w]] -> [x,y,z,t,w]
            out[obj_name] = flattened_obj_state
        return out
    
    def parse_st(self,st:np.ndarray)-> Dict[str,np.ndarray]:    
        out = {}
        for tup in self.st_legend:
            component_range, component_name = tup
            start = component_range[0]
            stop = component_range[1]
            out[component_name] = st[start:stop+1]
        return out
    
    def print_state(self,st):
        
        st_parsed = self.parse_st(st)
        print('s(t):')
        for item_name, item_st  in st_parsed.items():
            if 'mppi' in item_name:
                print('skipped mppi policy printing')
                continue
            print(f'{item_name}: {item_st}')
    
    @abstractmethod 
    def _select_action(self,st:torch.Tensor, *args, **kwargs)-> Tuple[int,dict]:
        """agent action selection - for implementention"""
        pass
    
    def select_action(self,st:torch.Tensor, *args, **kwargs):
        """
        Given state s(t) return action a(t) and its index.
        If training: select epsilon greedy. Else: select greedy action.
        
        Args:
            st (torch.Tensor): s(t)

        Returns:
            
        """
        ans = self._select_action(st, *args, **kwargs)
        action_idx:int = ans[0]
        action: Any = self.action_space[action_idx]
        meta_data:dict = ans[1]
        return action_idx, action, meta_data 
    
    @abstractmethod
    def _load(self, checkpoint:dict) -> None:
        pass 
    
    def load(self, checkpoint_path:str) -> Any:
        
        """ loading contents from checkpoint model, to "hot start" from a pre trained model.
        Returns:
            _type_: _description_
        """
        
        checkpoint = torch.load(checkpoint_path)
        self.set_training_episodes_done(checkpoint['training_episodes_done'])
        self.set_training_steps_done(checkpoint['training_steps_done'])
        self.set_test_episodes_done(checkpoint['test_episodes_done'])
        self.set_test_steps_done(checkpoint['test_steps_done'])
        self._load(checkpoint)
        
        # color_print(f'Model loading completed! checkpoint details:\n{checkpoint}')
        
        return checkpoint 
    
    @abstractmethod
    def _get_items_to_save(self, *args, **kwargs)->dict:
        pass 
    
    def save(self, checkpoint_file_path, *args, **kwargs):
        
        """ 
        saves a partially trained model with relevant info so training from the same state could resume      
        """ 
        training_episodes_done = self.get_training_episodes_done()
        training_steps_done = self.get_training_steps_done()
        test_episodes_done = self.get_test_episodes_done()
        test_steps_done = self.get_test_steps_done()
        
        # assert training_episodes_done > 0 # assert 'training_episodes_done' in kwargs and kwargs['training_episodes_done'] > 0
        # assert training_steps_done > 0  # assert 'training_steps_done' in  kwargs and kwargs['training_steps_done'] > 0        
        
        base_items = {
            'training_episodes_done': training_episodes_done,
            'training_steps_done': training_steps_done,
            'test_episodes_done': test_episodes_done,
            'test_steps_done': test_steps_done,
            
        }
        
        agent_specific_items = self._get_items_to_save(*args, **kwargs)
        items_to_save = {**base_items, **agent_specific_items} # syntax for merging two dicts
        torch.save(items_to_save, checkpoint_file_path)
    
        
    # @abstractmethod    
    # def _optimize(self, *args, **kwargs) -> dict:
    #     """optimization function of agent, to update network weights.
    #     Returns optimization meta data
        
    #     for example:
    #     self.train_suit.optimize(self.completed_optimization_steps_cntr)"""
        
    # def optimize(self,*args, **kwargs) -> dict:
    #     """a wrapper for the per-agent-implemented optimization step.
    #     Returns optimization meta data
    #     """
        
    #     optimization_meta_data = self._optimize(*args, **kwargs)
    #     self.completed_optimization_steps_cntr += 1
    #     return optimization_meta_data
    
        
    def flatten_sorted_coll_objs_states(self, col_obj_st_flat_states):
        """ 
        Flatten collision object locs (participating and not) to ndarray of all objects states
        Returns:

        """
        out = np.array([]) # one long state, will contain sequenced coll objs states (participating and not) sorted by their handle
        for obj_name in self.all_coll_obj_names_sorted:
            out= np.append(out, col_obj_st_flat_states[obj_name]) # np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]]) - >array([1, 2, 3, ..., 7, 8, 9])
        return out
    
    
    def _set_state_in_system(self, prev_at_idx,pi_mppi_means,pi_mppi_covs,robot_dof_positions_gym,robot_dof_velocities_gym,goal_pose_gym,completed_milestones, t):
        # Update the current state with the new components (only for components which can be changed)
        if 'prev_action_idx' in self.st_componentes_ordered:
            self.current_st['prev_action_idx'] = np.array([prev_at_idx]) 
        if 'pi_mppi_means' in self.st_componentes_ordered:
            if self.max_horizon == self.pi_mppi_means_horizon_in_st:
                self.current_st['pi_mppi_means'] = self._pad_pi_with_zeros(pi_mppi_means) # pi_mppi_means.flatten
            else:
                self.current_st['pi_mppi_means'] = pi_mppi_means[:self.pi_mppi_means_horizon_in_st]
        if 'pi_mppi_covs' in self.st_componentes_ordered:
            self.current_st['pi_mppi_covs'] = pi_mppi_covs # pi_mppi_means.flatten
        if 'robot_dofs_positions' in self.st_componentes_ordered:
            self.current_st['robot_dofs_positions'] = robot_dof_positions_gym
        if 'robot_dofs_velocities' in self.st_componentes_ordered:
            self.current_st['robot_dofs_velocities'] = robot_dof_velocities_gym
        if 'goal_pose' in self.st_componentes_ordered:
            self.current_st['goal_pose'] = goal_pose_gym
        if 'ee_err_milestones' in self.st_componentes_ordered: # milestones counter
            self.current_st['ee_err_milestones'] = completed_milestones 
        if 't' in self.st_componentes_ordered:
            self.current_st['t'] = t
    
    def compose_state_vector(self, robot_dof_positions_gym: np.ndarray, robot_dof_velocities_gym:np.ndarray, goal_pose_gym:np.ndarray, pi_mppi_means: np.ndarray, pi_mppi_covs:np.ndarray, completed_milestones_new:np.ndarray, t:int, prev_at_idx=-1) -> np.ndarray:
        """ given components of state, return encoded (flatten) state

        Args:
            st (_type_): _description_
        """
        
        
        self._set_state_in_system(prev_at_idx,pi_mppi_means,pi_mppi_covs,robot_dof_positions_gym,robot_dof_velocities_gym,goal_pose_gym,completed_milestones_new,t)
        # Vectorize the state components in the order of the state representation configuration
        ordered_st_components = [self.current_st[key].flatten() for key in self.st_componentes_ordered]
        current_st_vectorized = np.concatenate(ordered_st_components)    
        assert current_st_vectorized.ndim == 1 # TODO: this is verifying the shape is of a vector (shape is (n,)) and not of a matrix (shape is (1,n)).Remove it when convined its ok
        return current_st_vectorized
        
    
    def calc_component_size_in_state(self, component_name):
        """returns the number of cells in the whole state vector which the component takes """
        start_loc, end_loc = self.component_to_location_dict[component_name]
        return end_loc - start_loc + 1  
    
    def make_modified_state(self, base_state:np.ndarray, component_to_modify:str, new_component_val:np.ndarray):
        """makes a copy of base_state (flatten 1d state), modified only the the indices of the component named "component_to_modify", with the "new_component_val"  """
        modified_state = copy.copy(base_state)
        new_val_len = len(new_component_val)
        actual_len = self.calc_component_size_in_state(component_to_modify)
        assert new_val_len == actual_len   
        start_loc, end_loc = self.component_to_location_dict[component_to_modify]
        modified_state[0][start_loc: end_loc + 1] = torch.tensor(new_component_val) 
        return modified_state 
    
    def get_states_legend(self)->List[tuple]:
        """ 
        human readible legend for state representation
        """
        out = []
        start_idx = 0
        for i in range(len(self.st_componentes_ordered)):
            if self.st_componentes_ordered[i] != 'coll_objs':
                component_name = self.st_componentes_ordered[i]
                component_len_in_st = self.st_componentes_ordered_dims[i]
                end_idx = start_idx + component_len_in_st - 1 # inclusive
                out.append(((start_idx, end_idx),component_name))
                print(component_name, (start_idx, end_idx))
                start_idx = end_idx + 1
            else:
                for j in range(len(self.all_coll_obj_names_sorted)):
                    component_name = self.all_coll_obj_names_sorted[j]
                    component_len_in_st = len(self.col_obj_s0_flat_states[component_name]) # obj shape
                    end_idx = start_idx + component_len_in_st - 1 # inclusive
                    out.append(((start_idx, end_idx),component_name))
                    print(component_name, (start_idx, end_idx))
                    start_idx = end_idx + 1
        return out
                    
    def get_component_with_range_list(self):
        """ returns a list l where l[i] = (ith component name, (ith component start idx), ith component end idx))"""
        ans = []
        component_start_idx = 0
        for i in range(len(self.st_componentes_ordered)):
            component_name = self.st_componentes_ordered[i]
            component_end_idx = component_start_idx + self.st_componentes_ordered_dims[i] - 1 # inclisive    
            component_location = (component_start_idx, component_end_idx) # start index inclusive to end index inclusive 
            ans.append((component_name, component_location))
            component_start_idx = component_end_idx + 1 # move next component start index to next entry 
        # print(f'debug ans = {ans}')
        return ans
    
    def get_loc_of_component_in_state(self, component_name):
        """ returns start index (inclusive) to end index (inclusive) int he oredered flatten state vector"""
        return self.component_to_location_dict[component_name]
    
    def compute_reward(self, pos_err, rot_err, contact_detected, step_duration, goal_state, ts_cur, ts_max)-> Tuple[np.float32,dict]:
        """ A weighted sum of reward terms
        Returns:
            a. np.float64: total reward of the transition from s(t) to s(t+1).
            b. meta data
        """
        rlpt_cfg = GLobalVars.rlpt_cfg
        if rlpt_cfg is None:
            exit()
            
        # an error border which below it, we declare a state as a goal state:
        goal_pos_thresh_dist = rlpt_cfg['agent']['goal_test']['goal_pos_thresh_dist']
        goal_rot_thresh_dist = rlpt_cfg['agent']['goal_test']['goal_rot_thresh_dist']
        
        reward_config = rlpt_cfg['agent']['reward']
        step_dur_w = reward_config['step_dur_w']
        safety_w = reward_config['safety_w']
        pose_w = reward_config['pose_reward']['w']
        assert_positive = [pose_w, safety_w, step_dur_w, goal_pos_thresh_dist,goal_rot_thresh_dist]
        for arg in assert_positive:
            arg_name = f'{arg=}'.split('=')[0]
            assert arg >= 0, BadArgumentUsage(f"argument {arg_name} must be positve, but {arg} was passed.")

        # checking if the goal is reached (if the ee is close enough to the goal position and orientation)
        
        if reward_config['pose_reward']['use']: # if giving reward for goal reaching
            pose_reward = pose_w * int(goal_state) # positive only if in goal pose   
        else:
            pose_reward = 0 
        
        if self.is_milestone_cnt_changed: # achieved new milestone
            newly_achieved_milestone_index = int(self.current_st['ee_err_milestones']-1)
            milestone_reward = self.pose_err_milestones_reward_cfg['milestones'][newly_achieved_milestone_index][2]
            pose_reward += milestone_reward
        
        if reward_config['pose_reward']['ep_tail_err_reward']['use']:
            first_ts_in_tail = ts_max - len(self.ep_tail_err_reward_weights) 
            if ts_cur >= first_ts_in_tail:
                loc_in_arr = ts_cur - first_ts_in_tail # offset to array beginning
                w = self.ep_tail_err_reward_weights[loc_in_arr] * reward_config['pose_reward']['ep_tail_err_reward']['w']
                ep_tail_err_reward = - w * (pos_err + rot_err)
                pose_reward += ep_tail_err_reward 
                print(f'debug ts = {ts_cur}, max_ts = {ts_max}, w = {w}, ep_tail_r = {ep_tail_err_reward}',)

            
            
            
            
        safety_reward = safety_w * - int(contact_detected)                
        if reward_config['time_reward'] == 'linear':
            step_duration_reward = step_dur_w * - step_duration
        else: # binary, penalty of 1 for every step
            step_duration_reward = -1

        total_reward = pose_reward + safety_reward + step_duration_reward
        reward_metadata = {'pose_r_weighted': pose_reward, 'safety_r_weighted': safety_reward, 'dur_r_weighted': step_duration_reward}
        return np.float32(total_reward), reward_metadata

    def action_space_info(self):
        
        # deep copy and flattening of the action space 
        action_space_flatten = copy.deepcopy(self.action_space)
        for i in range(len(action_space_flatten)):
            action_space_flatten[i] = copy.deepcopy(action_space_flatten[i])
            action_space_flatten[i]['mpc_params'].update(action_space_flatten[i]['cost_weights'])
            action_space_flatten[i]['mpc_params']['goal_pose'] = tuple(action_space_flatten[i]['mpc_params']['goal_pose'])            
            action_space_flatten[i] = action_space_flatten[i]['mpc_params']
 
        df = pd.DataFrame.from_records(action_space_flatten) # rows = actions, columns = action dofs (features) 
        # Find columns where all values along rows are equal (action features that are the same over all actions)
        equal_columns = df.columns[df.nunique() == 1]
        df_equality = df[equal_columns][:1:] # equality between actions
 
        # Drop the columns with all equal values (more these action features to stay just with varying features)
        df_diffs = df.drop(equal_columns, axis=1) # differences between actions
        
        shared_params_all_actions:dict = df_equality.to_dict(orient='records')[0] # shared params between all actions
        different_params_by_action_inx:list = df_diffs.to_dict(orient='records') # l[i] a dict of the ith action, containing the unique assignment of this action to action features with 2 or more options  
        return shared_params_all_actions, different_params_by_action_inx 
    
    def _pad_pi_with_zeros(self, mppi_pi_means_cur_h:np.ndarray) -> np.ndarray:
        """ adding (max H - curr H) steps to policy with action = [0,0,0,0,0,0,0] (zero at each dof) for padding 

        Args:
            mppi_pi_means_cur_h (np.ndarray): 

        Returns:
            _type_: _description_
        """
        
        cur_pi_horizon = mppi_pi_means_cur_h.shape[0]
        rows_to_pad = self.max_horizon - cur_pi_horizon
        result = mppi_pi_means_cur_h
        if rows_to_pad > 0: # current policy is for H smaller than max H 
            trailing_zeros = np.zeros((rows_to_pad, 7))
            result = np.vstack((mppi_pi_means_cur_h, trailing_zeros))
        else:
            assert rows_to_pad == 0, 'bug'
        
        return result
           
    def _get_max_h(self):
        cur_max = 0
        for action in self.action_space:
            cur_max = max(cur_max, action['mpc_params']['horizon'])
        return cur_max
    
    def set_episodes_done(self, episodes_done):
        if self.training_mode:
            self.set_training_episodes_done(episodes_done)
        else:
            self.set_test_episodes_done(episodes_done)        
    
    def set_steps_done(self, steps):
        if self.training_mode:
            self.set_training_steps_done(steps)
        else:
            self.set_test_steps_done(steps)       
    
    def get_episodes_done(self):
        if self.training_mode:
            return self.get_training_episodes_done()
        else:
            return self.get_test_episodes_done()
        
        
    def get_steps_done(self):
        if self.training_mode:
            return self.get_training_steps_done()
        else:
            return self.get_test_steps_done()
        
            
    def set_training_episodes_done(self, training_episodes_done):
        assert training_episodes_done > 0
        self.trainig_episodes_done = training_episodes_done
    
    
    def get_training_episodes_done(self):
        return self.trainig_episodes_done
    
    def set_test_episodes_done(self, episodes_done):
        self.test_episodes_done = episodes_done
        
    def get_test_episodes_done(self):    
        return self.test_episodes_done
    
    def set_training_steps_done(self, training_steps_done):
        assert training_steps_done > 0
        self.trainig_steps_done = training_steps_done
        
    def get_training_steps_done(self):
        return self.trainig_steps_done
    
    def set_test_steps_done(self, steps_done):
        self.test_steps_done = steps_done
        
    def get_test_steps_done(self):
        return self.test_steps_done
    
    @abstractmethod
    def _training_step_post_ops(self,*args, **kwargs) -> Any:
        pass    
    
    @abstractmethod
    def _training_episode_post_ops(self,*args, **kwargs) -> Any:
        pass    
    @abstractmethod
    def _test_step_post_ops(self,*args, **kwargs) -> Any:
        pass
    def post_episode_ops(self,*args, **kwargs)->Any:
        
        logging_info = kwargs["episode_logging_info"]
        # print(f'debug 2: {len(logging_info["t_ep"])}, {len(logging_info["at_id"])}')
        
        if self.etl_logging:             
            special_labels = kwargs['special_labels'] if 'special_labels' in kwargs else []
            self.update_etl(kwargs['etl_file_path'], kwargs['episode_logging_info'],special_labels)

        self.set_episodes_done(self.get_episodes_done() + 1) # update episode cntr    
        if self.training_mode:
            ans = self.training_episode_post_ops(*args, **kwargs)
        else:
            ans = self.test_episode_post_ops(*args, **kwargs)
        return ans
    
    def post_step_ops(self, *args, **kwargs) -> Any:
        
        # update counter
        self.set_training_steps_done(self.get_training_steps_done() + 1)
    
        # any extra inheriting class ops
        if self.training_mode:
            ans = self._training_step_post_ops(*args, **kwargs)
        else:
            ans = self._test_step_post_ops(*args, **kwargs)
        return ans
        
    
    def training_episode_post_ops(self, checkpoint_file_path, *args, **kwargs) -> Any:
        
        # # update counter
        # self.set_training_episodes_done(self.get_training_episodes_done() + 1)
        
        # save updated info
        self.save(checkpoint_file_path)           
        
                
        # any extra inheriting class ops
        ans = self._training_episode_post_ops(*args, **kwargs)
        
        
        return ans
    
    def test_episode_post_ops(self,checkpoint_file_path, *args, **kwargs) -> Any:
        

        # save updated info
        self.save(checkpoint_file_path)           
        
                
        # any extra inheriting class ops
        # ans = self._test_episodes(*args, **kwargs)
        ans = None
        
        return ans
    
    def calc_obs_dim(self) -> int:
        """
        calculates observation space length

        """
        return self.st_dim
    
    def calc_action_dim(self) -> int:
        """
        calculates action  action space length
        
        """
        return self.at_dim
    
    def _calc_updated_milestones_status(self,pos_err, rot_err):
        """updating completed milestones count in new state

        Args:
            pos_err (_type_): _description_
            rot_err (_type_): _description_
        """
        def is_milestone_completed(pos_err, rot_err, milestone_index):
            milestone_list = self.pose_err_milestones_reward_cfg['milestones']
            next_milestone_to_achieve = milestone_list[milestone_index]
            pos_passed = pos_err < next_milestone_to_achieve[0]
            rot_passed = rot_err < next_milestone_to_achieve[1]
            if pos_passed and rot_passed: # milestone completed                
                return True
            return False

            
            
        completed_milestones_prev_st_cnt:np.ndarray = self.current_st['ee_err_milestones'] # achieved milestones count from prev state
        next_milestone_to_achieve_idx = int(completed_milestones_prev_st_cnt) 
        n_milestones = len(self.pose_err_milestones_reward_cfg['milestones']) # all milestones
        completed_all_milestones = completed_milestones_prev_st_cnt == n_milestones
        
        achieved_new_milestone = not completed_all_milestones and is_milestone_completed(pos_err, rot_err, next_milestone_to_achieve_idx)  
        updated_achieved_num:np.ndarray = completed_milestones_prev_st_cnt
        self.is_milestone_cnt_changed = False
        if achieved_new_milestone:
            updated_achieved_num += 1
            self.is_milestone_cnt_changed = True # so reward function could know
        return updated_achieved_num
        
        

    def calc_state(self, mpc, robot_handle, gymapi_state_all, sniffer, prev_action_idx, t) -> np.ndarray:
        # rlpt - compute the state you just moved to (next state, s(t+1)) 
        
        
        robot_dof_states_gym = mpc.gym.get_actor_dof_states(mpc.env_ptr, robot_handle, gymapi_state_all)# gymapi.STATE_ALL) # TODO may need to replace by 
        robot_dof_positions_gym: np.ndarray = robot_dof_states_gym['pos'] 
        robot_dof_vels_gym: np.ndarray =  robot_dof_states_gym['vel']
        goal_ee_pose_gym = mpc.get_body_pose(mpc.obj_body_handle, "gym") # in gym coordinate system
        goal_ee_pose_gym_np = pose_as_ndarray(goal_ee_pose_gym)        
        pi_mppi_means, pi_mppi_covs = sniffer.get_current_mppi_policy() 
        pi_mppi_means_np = torch_tensor_to_ndarray(pi_mppi_means)
        pi_mppi_covs_np = torch_tensor_to_ndarray(pi_mppi_covs).flatten() # [1,7] to [7]
        
        state_ee_pose_gym = mpc.get_body_pose(mpc.ee_body_handle, "gym")
        goal_ee_pose_gym = mpc.get_body_pose(mpc.obj_body_handle, "gym")
        ee_pos_error = pos_error(state_ee_pose_gym.p, goal_ee_pose_gym.p) # end effector position error (s(t+1))
        ee_rot_error = rot_error(state_ee_pose_gym.r, goal_ee_pose_gym.r)  # end effector rotation error (s(t+1))   
        completed_milestones_new = self._calc_updated_milestones_status(ee_pos_error,ee_rot_error)
        
        t = np.array([t])
        # compose state vector and update current state representation
        state_np = self.compose_state_vector(robot_dof_positions_gym, robot_dof_vels_gym, goal_ee_pose_gym_np, pi_mppi_means_np, pi_mppi_covs_np, completed_milestones_new,t, prev_action_idx) # converting the state to a form that agent would feel comfortable with
        
        return state_np
    
     
    def check_for_termination(self, sniffer, state_ee_pose_gym, goal_ee_pose_gym, goal_test_cfg):
        ee_pos_error = pos_error(state_ee_pose_gym.p, goal_ee_pose_gym.p) # end effector position error (s(t+1))
        ee_rot_error = rot_error(state_ee_pose_gym.r, goal_ee_pose_gym.r)  # end effector rotation error (s(t+1))   
        contact_detected = sniffer.is_contact_real_world or sniffer.is_self_contact_real_world
        is_goal_state = goal_test(ee_pos_error, ee_rot_error, goal_test_cfg) # rlpt_cfg['agent']['goal_test']
        terminated = is_goal_state or contact_detected # reached a terminal state (of environment)
        
        return terminated, is_goal_state, contact_detected, ee_pos_error, ee_rot_error
        
        