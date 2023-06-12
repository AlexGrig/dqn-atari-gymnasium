import yaml
import inspect

class NestedDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        keys_to_modify = []
        for key,value in self.items():
            if isinstance(value, dict):
                keys_to_modify.append(key)
        
        for mod_key in keys_to_modify:
            self[mod_key] = NestedDict(self[mod_key])
            
    def _get_elements(self, keys_filter_list=None):
        elements = {}

        def _traverse(dictionary, parent_key=''):
            for key, value in dictionary.items():
                new_key = key #f"{parent_key}.{key}" if parent_key else key
                if new_key in elements:
                    raise ValueError(f'The config dictionary has duplicate keys: {new_key}')
                    
                if isinstance(value, dict):
                    _traverse(value, parent_key=new_key)
                else:
                    if keys_filter_list is not None:
                        if new_key in keys_filter_list:
                            elements[new_key] = value
                    else:
                        elements[new_key] = value    
        _traverse(self)
        return elements
    def get_all_elements(self,):
        return self._get_elements(keys_filter_list=None)
    
    def filter_elements_with_func_named_args(self, func):
        
        func_args = inspect.getfullargspec(func).args
        func_default_vals = inspect.getfullargspec(func).defaults
        #import pdb; pdb.set_trace()
        func_default_args = func_args[-len(func_default_vals):]
    
        return self._get_elements(keys_filter_list=func_default_args)
    
    def __add__(self, other):
        other_dict = dict(other)
        result = dict(self)
        result.update(other_dict)
        return result

    def __radd__(self, other):
        return self.__add__(other)
    
    def __repr__(self,):
        return f"NestedDict({super().__repr__()})"
        
def read_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ValueError("Yaml error - check yaml file")
            
        return NestedDict(data)
