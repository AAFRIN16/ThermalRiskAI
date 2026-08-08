import os
import json
import logging
import firebase_admin
from firebase_admin import credentials, auth

logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK app if not already initialized
if not firebase_admin._apps:
    try:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        hf_secret = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")

        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized using local credentials file: %s", cred_path)
        elif hf_secret:
            parsed_json = json.loads(hf_secret)
            cred = credentials.Certificate(parsed_json)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized using Hugging Face Secret.")
        else:
            firebase_admin.initialize_app()
            logger.info("Firebase Admin SDK initialized using default credentials.")
    except Exception as e:
        logger.warning("Firebase Admin SDK initialization warning: %s", e)


def verify_firebase_token(id_token: str) -> dict:
    """
    Verifies a Firebase ID token.
    Returns decoded token dictionary containing user info:
    - uid
    - email
    - name
    - picture (photoURL)
    """
    if not id_token:
        raise ValueError("Token is required")

    # Support development/testing mock token
    if id_token.startswith("dev_") or id_token == "test_token":
        dev_uid = id_token if id_token.startswith("dev_") else "dev_user_001"
        return {
            "uid": dev_uid,
            "email": f"{dev_uid}@example.com",
            "name": f"Dev User ({dev_uid})",
            "picture": "https://via.placeholder.com/150",
        }

    try:
        decoded_token = auth.verify_id_token(id_token, check_revoked=False)
        uid = decoded_token.get("uid")
        email = decoded_token.get("email", "")
        name = decoded_token.get("name") or decoded_token.get("email", "").split("@")[0] or "ThermalRisk User"
        picture = decoded_token.get("picture", "")

        return {
            "uid": uid,
            "email": email,
            "name": name,
            "picture": picture,
            "auth_time": decoded_token.get("auth_time"),
        }
    except Exception as e:
        logger.error("Failed to verify Firebase token: %s", e)
        raise ValueError(f"Invalid or expired Firebase ID token: {str(e)}")
