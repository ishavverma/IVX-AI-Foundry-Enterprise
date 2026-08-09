from __future__ import annotations

import re
import requests

BASE_URL = "http://127.0.0.1:5000"

def csrf(html: str) -> str:
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)

session = requests.Session()
login_page = session.get(f"{BASE_URL}/login")
print("login_page", login_page.status_code, login_page.url)
if login_page.status_code != 200:
    raise RuntimeError(f"Chat probe login page unavailable: {login_page.status_code}")
login = session.post(
    f"{BASE_URL}/login",
    data={
        "username": "training_user_20260808",
        "password": "TrainUser@12345",
        "csrf_token": csrf(login_page.text),
    },
)
if "/login" in login.url:
    raise RuntimeError(f"Chat probe login failed: {login.status_code} {login.url}")
chat_page = session.get(f"{BASE_URL}/chat/")
session_match = re.search(r"chatStudio\('([0-9a-f-]+)'", chat_page.text)
if not session_match:
    print("Chat page snippet:", chat_page.text[:500])
    raise RuntimeError("Chat session ID not found")
csrf_token = csrf(chat_page.text)
response = session.post(
    f"{BASE_URL}/chat/send",
    headers={"X-CSRFToken": csrf_token},
    json={"session_id": session_match.group(1), "message": "Explain the hybrid Mamba-2 SSM and Transformer attention architecture in AFE-Base-7B."},
)
print("Status code:", response.status_code)
print("Response JSON:", response.json())

