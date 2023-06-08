from pathlib import Path
import csv

class ActionLogger():
    def __init__(self, action_selections, log_file, action_log_frequency, erase_existing=True):
        """
        action_selection (dict): mapping from int to actions in the environment.
        """
        self.action_selections = action_selections
        self.log_file = Path(log_file).absolute()
        self.action_log_frequency = action_log_frequency
        
        if erase_existing:
            if log_file.exists():
                log_file.unlink(missing_ok=True)
                
        log_dir = log_file.parent
        if not log_dir.exists():
            log_dir.mkdir()
        
        
    def log_action(self, step, action):
        self.action_selections[int(action)] += 1.0/self.action_log_frequency
        if step % self.action_log_frequency == 0:
            with self.log_file.open(mode='a', buffering=1) as ff:
                writer = csv.writer(ff)
                writer.writerow([step,] + self.action_selections)
            self.action_selections = [0 for _ in range(len(self.action_selections))]

class Logger():
    def __init__(self, log_file, erase_existing=True):
        """
        
        """
        self.log_file = Path(log_file).absolute()
        self._header_written = False
        self._keys = None
        if erase_existing:
            self._erase_existing_log(self.log_file)
            
        log_dir = log_file.parent
        if not log_dir.exists():
            log_dir.mkdir()
        
    def _erase_existing_log(self, log_file):
        if log_file.exists():
            log_file.unlink(missing_ok=True)
        
    def _write_header(self, field_names_list):
        with self.log_file.open(mode='w', buffering=1) as ff:
            writer = csv.writer(ff)
            writer.writerow(field_names_list)
    
    def log(self, log_dict):
        if not self._header_written:
            keys = list(log_dict.keys())
            self._keys = keys
            self._write_header(keys)
            self._header_written = True
            
        with self.log_file.open(mode='a', buffering=1) as ff:
            writer = csv.writer(ff)
            writer.writerow(list(log_dict.values()))