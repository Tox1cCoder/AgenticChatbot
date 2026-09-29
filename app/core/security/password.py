"""
Password hashing utilities
"""

import bcrypt

# bcrypt only ever reads the first 72 bytes. bcrypt 5 raises ValueError instead
# of truncating, so every longer password became a 500 at signup and at login.
BCRYPT_MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """Check ``password`` against a stored bcrypt hash.

    Truncates to the 72 bytes bcrypt compares. Hashes stored before bcrypt 5
    were made from the silently truncated password, so this keeps those users
    able to log in with the password they actually typed; new signups cannot
    exceed the limit (``UserCreate`` rejects them).
    """
    password_bytes = password.encode("utf-8")[:BCRYPT_MAX_PASSWORD_BYTES]
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_bytes)
