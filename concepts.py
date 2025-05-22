from typing import List
from pydantic import BaseModel, Field


class Concepts(BaseModel):

    """Pydantic model for representing a list of concepts."""

    concepts_list: List[str] = Field(description="List of concepts")