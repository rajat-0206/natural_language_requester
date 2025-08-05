from dataclasses import dataclass
from .datamodel import DataModel

@dataclass
class MissingFieldRequestModel(DataModel):
    missing_fields: list[str]
    current_params: dict
    matched_api: dict
    request_method: str
    action: str
    user_input: str