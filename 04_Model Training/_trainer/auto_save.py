import torch
import json
from pathlib import Path

CONFIG_FILE_NAME = 'config.json'

class AutoSave:
    def __init__(self, model, path):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True, parents=True)
      
        self.model_name = model.name_or_path
        if hasattr(model, '_delta_module'):
            self.delta_module = model._delta_module
        else:
            self.model = model
        self._save_config()
                
    def _save_config(self):
        config = {
            'model_name': self.model_name,
        }
        if self.has_delta:
            config['peft_config'] = self.delta_module.peft_config()
        with open(self.path / CONFIG_FILE_NAME, 'w') as f:
            json.dump(config, f)
                
    @property
    def has_delta(self):
        return hasattr(self, 'delta_module')
            
    def save(self, name):
        if self.has_delta:
            state_dict = self.delta_module.peft_state_dict()
        else:
            state_dict = self.model.state_dict()
        torch.save(state_dict, self.path / f'{name}.pt')
        
    def load(self, name):
        with open(self.path / CONFIG_FILE_NAME, 'r') as f:
            config = json.load(f)
        state_dict = torch.load(self.path / f'{name}.pt')
        self.delta_module.load_peft(config=config['peft_config'], state_dict=state_dict)