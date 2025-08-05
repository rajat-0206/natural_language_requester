from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

class DataModel(ABC):
    def to_dict(self):
        return self.__dict__
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self):
        return self.to_json()