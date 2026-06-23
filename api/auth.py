from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
from database import users_collection

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

router = APIRouter()

SECRET_KEY = os.getenv("JWT_SECRET", "mindlens_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ---------------- SCHEMAS ---------------- #

class SignupModel(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginModel(BaseModel):
    email: EmailStr
    password: str

import bcrypt

def hash_password(password):
    if len(password) > 72:
        password = password[:72]
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain, hashed):
    try:
        return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False

def create_token(data: dict):
    expire = datetime.utcnow() + timedelta(days=7)
    data.update({"exp": expire})
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# ---------------- ROUTES ---------------- #

@router.post("/signup")
def signup(user: SignupModel):
    try:
        existing = users_collection.find_one({"email": user.email})
    except Exception as e:
        raise HTTPException(status_code=503, detail="Database is offline. Cannot signup right now.")

    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")

    try:
        hashed_pw = hash_password(user.password)
    except Exception as e:
        print("EXCEPTION IN HASHING:", e)
        raise HTTPException(status_code=500, detail="Internal server error during password hashing.")
        
    try:
        users_collection.insert_one({
            "name": user.name,
            "email": user.email,
            "password": hashed_pw,
            "created_at": datetime.utcnow()
        })
    except Exception as e:
        print("EXCEPTION IN SIGNUP:", e)
        raise HTTPException(status_code=503, detail="Database is offline. Cannot complete signup.")

    return {"message": "User created successfully"}

@router.post("/login")
def login(user: LoginModel):
    try:
        db_user = users_collection.find_one({"email": user.email})
    except Exception as e:
        raise HTTPException(status_code=503, detail="Database is offline. Cannot login right now.")

    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid email")

    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Wrong password")

    token = create_token({
        "user_id": str(db_user["_id"]),
        "email": db_user["email"]
    })

    return {
        "access_token": token,
        "token_type": "bearer"
    }

def get_current_user(token: str = Depends(oauth2_scheme)):

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("email")

        try:
            user = users_collection.find_one({"email": email})
        except Exception as e:
            print(f"Warning: DB offline during auth check: {e}")
            user = {
                "_id": payload.get("user_id", "offline_id"),
                "name": "Offline User",
                "email": email
            }

        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        return {
            "id": str(user.get("_id", payload.get("user_id"))),
            "name": user.get("name", "Offline User"),
            "email": user.get("email", email)
        }

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return current_user