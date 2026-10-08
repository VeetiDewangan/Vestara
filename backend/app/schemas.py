from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class WardrobeItemCreate(BaseModel):
    filename: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    colors: Optional[List[str]] = []
    tags: Optional[List[str]] = []

class WardrobeItemResponse(BaseModel):
    id: str
    filename: str
    image_url: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    colors: List[str]
    tags: List[str]
    created_at: datetime

    class Config:
        from_attributes = True # updated from orm_mode for pydantic v2
