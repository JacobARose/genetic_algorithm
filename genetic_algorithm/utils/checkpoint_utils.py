"""
Created 12/12/2020 by Jacob A Rose

[CheckpointManager]: Class for managing a set of checkpoints contained in a single root directory

TODO:

[Checkpoint]: Class meant to represent a single Checkpoint for a CheckpointManager to supervise. CheckpointManager will then directly handle a set of Checkpoint instances, for each task in a model's lifetime sequence.

[TaskSpecificCheckpointSubclasses]: Create a unique subclass for categories of tasks that require keeping track of different subsets of data/metadata

root:
    /path/to/root_dir

tasks:
    ./init/
    ./warmup/
    ./tune/
    ./pretext/
    ./train/

"""

import dataclasses
from typing import Optional
from pathlib import Path
import os
from contrastive_learning.data import stateful






@dataclasses.dataclass
class Checkpoint(stateful.Stateful):
    """[summary]

    Args:
        stateful ([type]): [description]
    """
    task_id: str
    root_dir: str
    checkpoint_files = ('checkpoint', '{task_id}.json', 'chkpt.index')

    def __init__(self,
                 task_id: str,
                 root_dir: str):
        super().__init__(task_id=task_id, root_dir=root_dir)


class CheckpointManager(stateful.Stateful):
    """ CheckPointManager Class
    - Interface for Reading/Writing/Locating valid Checkpoints on disk created previously by the same interface

    """    
    task_order = ('init', 'warmup', 'tune', 'pretext', 'train')
    checkpoint_files = ('checkpoint', '{task}.json', 'chkpt.index')
    
    
    def __init__(self, ckpt_root_dir: str):
        self.root_dir = ckpt_root_dir
    
    @classmethod
    def validate_checkpoint_dir(cls, ckpt_dir: str):
        """
        Verify that user can load the checkpoint located in ckpt_dir
        Return ckpt_dir if checkpoint contains valid files as specified in class attribute checkpoint_files,
        else: return None

        Args:
            ckpt_dir (str): [description]

        Returns:
            [type]: [description]
        """ 
        
        try:
            unvetted_task = Path(ckpt_dir).name
            if unvetted_task in ['warmup','tune']:
                unvetted_task = 'hp'                
                
            checkpoint_files = cls.checkpoint_files
            print(os.path.join(ckpt_dir, checkpoint_files[0]))
            print(os.path.join(ckpt_dir, checkpoint_files[1].format(task=unvetted_task)))
            print(os.path.join(ckpt_dir, checkpoint_files[2]))
            assert os.path.exists(os.path.join(ckpt_dir, checkpoint_files[0]))
            assert os.path.exists(os.path.join(ckpt_dir, checkpoint_files[1].format(task=unvetted_task)))
            assert os.path.exists(os.path.join(ckpt_dir, checkpoint_files[2]))
            return ckpt_dir
        except AssertionError as e:
            print(e)
            return None
        except Exception as e:
            print(e)
        
    @classmethod
    def find_latest_checkpoint(cls, 
                               root_dir: str='.', 
                               last_allowed: str=None
                               ) -> Optional[str]:
        """[summary]

        Args:
            root_dir (str, optional): [description]. Defaults to '.'.
            last_allowed (str, optional): [description]. Defaults to None.

        Returns:
            Optional[str]: [description]
        """                               
        ''' Search "root_dir" for subdirectories named for a valid task, inside of which belongs all relevant checkpoint assets/files/folders.
        
        
        '''
        previous_tasks = os.listdir(root_dir)
        
        valid_checkpoints = []
        for task in cls.task_order:
            if task in previous_tasks:
                unvetted_task_dir = os.path.join(root_dir, task)
                vetted_task_dir = cls.validate_checkpoint_dir(unvetted_task_dir)
                if isinstance(vetted_task_dir, str):
                    print(f'success: vetted_task_dir={vetted_task_dir}')
                    valid_checkpoints.append(vetted_task_dir)
                else:
                    print(vetted_task_dir)
                if task == last_allowed:
                    break
                    
        if len(valid_checkpoints)>0:
            latest_checkpoint = valid_checkpoints[-1]
            
            print(f'Found {len(valid_checkpoints)} valid checkpoints stored in root directory:\n\t{root_dir}')
            print('Available checkpoints include:\n', '\n'.join([ckpt_name for ckpt_name in valid_checkpoints]))
#             print('Available checkpoints include:\n', *[ckpt_name for ckpt_name in os.listdir(valid_checkpoints)])
            print(f'Returning the latest checkpoint in {Path(latest_checkpoint).name}')
            return latest_checkpoint
        
        print(f'Found no previous checkpoints in {root_dir}, initiating from scratch')
        return None