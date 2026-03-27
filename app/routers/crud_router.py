from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/items", tags=["CRUD"])

class Item(BaseModel):
    id: int
    name: str

items_db = []

@router.post("/")
def create_item(item: Item):
    items_db.append(item)
    return {"message": "Item created", "item": item}

@router.get("/")
def get_items():
    return items_db

@router.put("/{item_id}")
def update_item(item_id: int, updated_item: Item):
    for i, item in enumerate(items_db):
        if item.id == item_id:
            items_db[i] = updated_item
            return {"message": "Item updated", "item": updated_item}
    return {"error": "Item not found"}

@router.delete("/{item_id}")
def delete_item(item_id: int):
    for i, item in enumerate(items_db):
        if item.id == item_id:
            deleted = items_db.pop(i)
            return {"message": "Item deleted", "item": deleted}
    return {"error": "Item not found"}