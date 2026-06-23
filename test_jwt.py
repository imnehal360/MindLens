import requests

BASE_URL = "http://127.0.0.1:8000"

def test_jwt_flow():
    # 1. Register
    print("Registering...")
    res = requests.post(f"{BASE_URL}/signup", json={"name": "Test User", "email": "test_jwt_user@example.com", "password": "password123"})
    print(res.status_code, res.json())

    # 2. Login
    print("\nLogging in...")
    res = requests.post(f"{BASE_URL}/login", json={"email": "test_jwt_user@example.com", "password": "password123"})
    print(res.status_code, res.json())
    
    if res.status_code != 200:
        print("Login failed, aborting tests.")
        return
        
    token = res.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Test unauthenticated request to /mental-health-report
    print("\nTesting /mental-health-report WITHOUT token...")
    res_no_auth = requests.get(f"{BASE_URL}/mental-health-report")
    print(res_no_auth.status_code, res_no_auth.json())
    
    # 4. Test authenticated request to /mental-health-report
    print("\nTesting /mental-health-report WITH token...")
    res_auth = requests.get(f"{BASE_URL}/mental-health-report", headers=headers)
    print(res_auth.status_code, res_auth.json())

    # 5. Test authenticated request to /predict/fusion
    print("\nTesting /predict/fusion WITH token...")
    res_fusion = requests.post(f"{BASE_URL}/predict/fusion", data={"text": "I am feeling happy today."}, headers=headers)
    print(res_fusion.status_code, res_fusion.json())

if __name__ == "__main__":
    test_jwt_flow()
