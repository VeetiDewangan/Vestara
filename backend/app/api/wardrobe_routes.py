from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..database import get_db
from ..models import WardrobeItem
from ..schemas import WardrobeItemCreate, WardrobeItemResponse

router = APIRouter()

@router.post("/", response_model=WardrobeItemResponse)
def create_item(item: WardrobeItemCreate, db: Session = Depends(get_db)):
    # Convert lists to comma-separated strings for SQLite storage
    colors_str = ",".join(item.colors) if item.colors else ""
    tags_str = ",".join(item.tags) if item.tags else ""
    
    db_item = WardrobeItem(
        filename=item.filename,
        category=item.category,
        subcategory=item.subcategory,
        colors=colors_str,
        tags=tags_str
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    
    # Format response
    return WardrobeItemResponse(
        id=db_item.id,
        filename=db_item.filename,
        image_url=f"/api/v1/clothing/result/{db_item.filename}",
        category=db_item.category,
        subcategory=db_item.subcategory,
        colors=colors_str.split(",") if colors_str else [],
        tags=tags_str.split(",") if tags_str else [],
        created_at=db_item.created_at
    )

@router.get("/", response_model=List[WardrobeItemResponse])
def get_items(db: Session = Depends(get_db)):
    items = db.query(WardrobeItem).all()
    response = []
    for db_item in items:
        response.append(WardrobeItemResponse(
            id=db_item.id,
            filename=db_item.filename,
            image_url=f"/api/v1/clothing/result/{db_item.filename}",
            category=db_item.category,
            subcategory=db_item.subcategory,
            colors=db_item.colors.split(",") if db_item.colors else [],
            tags=db_item.tags.split(",") if db_item.tags else [],
            created_at=db_item.created_at
        ))
    return response

@router.delete("/{item_id}")
def delete_item(item_id: str, db: Session = Depends(get_db)):
    db_item = db.query(WardrobeItem).filter(WardrobeItem.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    db.delete(db_item)
    db.commit()
    return {"message": "Item deleted successfully"}
