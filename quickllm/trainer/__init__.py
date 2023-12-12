'''Trainer
通用的Trainer定义在torch4keras中，这里定义NLP中独有的Trainer
'''

from torch4keras.trainer import *  # torch4keras>=0.1.2.post2
from quickllm.trainer.ppo_trainer import PPOTrainer
from quickllm.trainer.dpo_trainer import DPOTrainer