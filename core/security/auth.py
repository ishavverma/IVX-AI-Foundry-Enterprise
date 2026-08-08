"""
AFE — Authentication Service

Provides password hashing (bcrypt), JWT token generation / validation,
and user lookup helpers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt

from core.config.settings import get_settings
from core.database.engine import get_session
from core.database.models.user import User

logger = logging.getLogger(__name__)
_settings = get_settings()


class AuthService:
    """Stateless authentication utilities."""

    # ── Password Hashing ─────────────────────────

    @staticmethod
    def hash_password(plain: str) -> str:
        """Return a bcrypt hash of the plaintext password."""
        salt = bcrypt.gensalt(rounds=_settings.BCRYPT_ROUNDS)
        return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        """Check a plaintext password against a bcrypt hash."""
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

    # ── JWT Tokens ───────────────────────────────

    @staticmethod
    def create_access_token(user_id: str, extra_claims: dict | None = None) -> str:
        """Generate a short-lived JWT access token."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(minutes=_settings.JWT_ACCESS_TOKEN_EXPIRES_MINUTES),
            "type": "access",
        }
        if extra_claims:
            payload.update(extra_claims)
        return jwt.encode(payload, _settings.SECRET_KEY, algorithm=_settings.JWT_ALGORITHM)

    @staticmethod
    def create_refresh_token(user_id: str) -> str:
        """Generate a long-lived JWT refresh token."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(days=_settings.JWT_REFRESH_TOKEN_EXPIRES_DAYS),
            "type": "refresh",
        }
        return jwt.encode(payload, _settings.SECRET_KEY, algorithm=_settings.JWT_ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> Optional[dict]:
        """Decode and validate a JWT token.  Returns the payload dict or None."""
        try:
            return jwt.decode(token, _settings.SECRET_KEY, algorithms=[_settings.JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as exc:
            logger.warning("Invalid token: %s", exc)
            return None

    # ── User Helpers ─────────────────────────────

    @staticmethod
    def authenticate(username_or_email: str, password: str) -> Optional[User]:
        """Validate credentials and return the User or None."""
        with get_session() as session:
            user = (
                session.query(User)
                .filter(
                    (User.username == username_or_email) | (User.email == username_or_email)
                )
                .first()
            )
            if user and AuthService.verify_password(password, user.password_hash):
                if not user.is_active:
                    logger.warning("Inactive user attempted login: %s", user.username)
                    return None
                return user
            return None

    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[User]:
        """Fetch a user by primary key."""
        with get_session() as session:
            return session.get(User, user_id)

    @staticmethod
    def register_user(
        username: str,
        email: str,
        password: str,
        display_name: str = "",
        is_superadmin: bool = False,
    ) -> User:
        """Create a new user with a hashed password."""
        with get_session() as session:
            user = User(
                username=username,
                email=email,
                password_hash=AuthService.hash_password(password),
                display_name=display_name or username,
                is_superadmin=is_superadmin,
            )
            session.add(user)
            session.flush()
            logger.info("Registered user: %s (%s)", username, user.id)
            return user
