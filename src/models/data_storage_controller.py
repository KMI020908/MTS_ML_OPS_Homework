import os
import pathlib
import numpy as np


class DataStorage:

    def __init__(self, storage_volume, storage_del_frac, data_paths, relative_path="data"):
        self.storage_volume = storage_volume
        self.data_paths = data_paths
        self.relative_path = relative_path
        self.storage_del_frac = storage_del_frac

    def process_storage(self):
        for data in self.data_paths:
            dir_lst = os.listdir(os.path.join(self.relative_path, data['dir']))
            if len(dir_lst) > self.storage_volume:
                dir_lst = np.sort(dir_lst)
                last_file = int(self.storage_volume * self.storage_del_frac)
                for file in np.sort(dir_lst)[:last_file+1]:
                    if file != '.pathholder':
                        path = pathlib.Path(os.path.join(self.relative_path, data['dir'], file))
                        path.unlink()
