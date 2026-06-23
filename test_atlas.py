from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("MONGO_URI")


client = MongoClient(uri)

db = client["mindlens"]

collection = db["test"]

collection.insert_one({
    "name": "Nehal",
    "project": "MindLens"
})

print("Connected & Inserted Successfully")