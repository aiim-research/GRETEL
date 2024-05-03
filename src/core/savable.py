from abc import ABCMeta, abstractmethod
from datetime import timedelta
from os.path import exists
from time import sleep
from flufl.lock import Lock, TimeOutError

from src.core.configurable import Configurable
from src.utils.context import Context


class Savable(Configurable, metaclass=ABCMeta):
    def __init__(self, context: Context, local_config):
        super().__init__(context, local_config)
        self.load_or_create()

    @abstractmethod 
    def write(self):
        pass

    @abstractmethod 
    def read(self):
        pass

    @abstractmethod
    def create(self):
        pass

    def saved(self):
        return exists(self.context.get_path(self))

    def load_or_create(self, condition_ext=False):
        path = f'{self.context.get_path(self)}.lck'
        lifetime = timedelta(hours=self.context.lock_release_tout)
        timeout = timedelta(seconds=self.context.lock_timeout)
        
        while True:
            try:
                lock = Lock(path, lifetime, default_timeout=timeout)
                condition_ext = condition_ext if not lock.is_locked() else False
                with lock:
                    condition = condition_ext or not self.saved()
                    if condition:
                        self.context.logger.info(f"Creating: {self}")
                        self.create()
                        self.write()
                        self.context.logger.info(f"Saved: {self}")
                    else:
                        self.context.logger.info(f"Loading: {self}")
                        self.read()
            except TimeOutError:
                sleep(1)
                continue
            break
