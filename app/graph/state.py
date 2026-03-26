from typing import TypedDict, Optional, List

from pydantic import BaseModel, Field
from typing import Optional

class AgentState(TypedDict):
    query: str
    repo_url: str
    branch: Optional[str]
    filename: Optional[str]
    context: List[str]
    answer: str
    retry_count: int
