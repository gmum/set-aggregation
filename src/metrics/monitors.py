import os.path as path
import numpy as np


class Monitor:
    def __init__(self, folder):
        self.folder = folder
        self.monitor_vars = {}

    def add_variable(self, name, val):
        if name not in self.monitor_vars:
            self.monitor_vars[name] = []
        self.monitor_vars[name].append(val)

    def monitor_all(self, names, vals):
        for name, val in zip(names, vals):
            self.add_variable(name, val)

    def save(self):
        for name in self.monitor_vars.keys():
            outfile = path.join(self.folder, name+".npy")
            np.save(outfile, self.monitor_vars[name])

    def save_args(self, args, config_name):
        with open(path.join(self.folder, config_name), 'w') as fout:
            for arg, value in args.__dict__.items():
                fout.write('{} : {}\n'.format(arg, value))

    def save_session(self, session, saver, model_name):
        saver.save(session, path.join(self.folder, model_name))
