from argparse import Namespace

class ModelAPI(object):

    def __init__(self, config: Namespace):
        self._config = config