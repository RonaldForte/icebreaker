from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["CRUD"])


class User(BaseModel):
    id: int
    username: str


# A tiny in-memory store is enough to satisfy the CRUD requirement without adding DB complexity.
users_db = []


@router.post("/users")
def create_user(user: User):
    # Keep ids unique so the selected Streamlit user is stable across requests.
    if any(existing.id == user.id for existing in users_db):
        return {"error": "User with this id already exists"}
    users_db.append(user)
    return {"message": "User created", "user": user}


@router.get("/users")
def get_users():
    return users_db


@router.get("/users/{user_id}")
def get_user(user_id: int):
    # Linear lookup is fine here because the dataset is tiny and ephemeral.
    for user in users_db:
        if user.id == user_id:
            return user
    return {"error": "User not found"}


@router.delete("/users/{user_id}")
def delete_user(user_id: int):
    for i, user in enumerate(users_db):
        if user.id == user_id:
            deleted = users_db.pop(i)
            return {"message": "User deleted", "user": deleted}
    return {"error": "User not found"}
