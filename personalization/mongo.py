import os
from pathlib import Path
from dotenv import load_dotenv

class LazyCollection:
    def __init__(self, collection_name):
        self._collection_name = collection_name
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            from pymongo import MongoClient
            
            ROOT_DIR = Path(__file__).resolve().parent.parent
            load_dotenv(ROOT_DIR / ".env")
            uri = os.getenv("MONGO_URI")

            client = None
            try:
                if uri:
                    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
                    client.admin.command('ping')
                    print(f"MongoDB connected successfully using URI.")
                else:
                    raise Exception("No MONGO_URI provided in environment.")
            except Exception as e:
                print(f"Warning: Failed to connect to MongoDB URI. Falling back to local MongoDB on port 27017. Error: {e}")
                try:
                    local_uri = "mongodb://localhost:27017"
                    client = MongoClient(local_uri, serverSelectionTimeoutMS=3000)
                    client.admin.command('ping')
                    print("MongoDB connected successfully to local fallback: mongodb://localhost:27017")
                except Exception as local_err:
                    print(f"Error: Failed to connect to local MongoDB fallback: {local_err}")
                    raise ConnectionError("MongoDB is completely unreachable (both configured URI and local fallback).") from local_err

            db = client["mindlens"]
            self._collection = db[self._collection_name]
        return self._collection

    def __getattr__(self, name):
        return getattr(self._get_collection(), name)

    def __getitem__(self, name):
        return self._get_collection()[name]

# Collections
history_collection = LazyCollection("history")
users_collection = LazyCollection("users")
alerts_collection = LazyCollection("alerts")
predictions_collection = LazyCollection("predictions")