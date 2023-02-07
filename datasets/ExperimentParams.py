"""
@author Liu Lei
"""

class Params(object):
    def __init__(self):
        super().__init__()
        self.debug = False
        self.adap=True
        self.save_pt=True

        self.repeats = 10
        self.repeat=1
        self.data_type=None

        self.each_samples = 600
        self.batch_size = 1
        self.train_rate = 0.5
        self.epochs = 3
        self.lr = 0.001
        self.k = 2
        self.sample_len=None
        self.norm='IN'
        self.g=2
        self.dataset_name=None
        self.class_num=10

