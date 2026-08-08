from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class UserModel(BaseModel):
    uid: str = Field(..., description="Firebase User ID")
    name: str = Field(..., description="Full Name")
    email: str = Field(..., description="Email address")
    photoURL: Optional[str] = Field(default="", description="User avatar URL")
    createdAt: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Creation timestamp")
    lastLogin: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Last login timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "uid": "usr_123456",
                "name": "Jane Doe",
                "email": "jane@example.com",
                "photoURL": "https://lh3.googleusercontent.com/a/default-user",
                "createdAt": "2026-08-08T14:00:00Z",
                "lastLogin": "2026-08-08T14:30:00Z"
            }
        }
