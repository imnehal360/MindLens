from datetime import datetime
from personalization.mongo import history_collection

session_history = {}

def save_history(
    user_id,
    depression,
    anxiety,
    stress,
    risk,
    future_risk=None,
    status="stable"
):

    data = {
        "user_id": user_id,
        "depression": float(depression),
        "anxiety": float(anxiety),
        "stress": float(stress),
        "risk": float(risk),
        "future_risk": float(future_risk) if future_risk is not None else None,
        "status": status,
        "timestamp": datetime.utcnow()
    }

    try:
        history_collection.insert_one(data)
    except Exception as e:
        print(f"Warning: Could not save history to DB: {e}")
        if user_id not in session_history:
            session_history[user_id] = []
        session_history[user_id].append(data)


def get_last_7(user_id):

    try:
        data = list(
            history_collection.find(
                {"user_id": user_id},
                {"_id": 0}
            )
            .sort("timestamp", -1)
            .limit(7)
        )

        data.reverse()

        return data
    except Exception as e:
        print(f"Warning: Could not fetch history from DB: {e}")
        if user_id in session_history:
            return session_history[user_id][-7:]
        return []