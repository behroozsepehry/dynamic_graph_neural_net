from typing import List
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError
