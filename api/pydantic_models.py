from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    # LLAMA32_1B = "meta-llama/Llama-3.2-1B-Instruct"
    LLAMA32_1B = "llama3.2:1b"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.LLAMA32_1B)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int