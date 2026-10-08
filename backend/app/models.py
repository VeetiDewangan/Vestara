import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime
from .database import Base

class WardrobeItem(Base):
    __tablename__ = "wardrobe_items"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    filename = Column(String, nullable=False)
    category = Column(String, nullable=True)
    subcategory = Column(String, nullable=True)
    colors = Column(String, nullable=True) # Stored as comma-separated
    tags = Column(String, nullable=True)   # Stored as comma-separated
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
