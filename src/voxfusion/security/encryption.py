"""Optional output file encryption using Fernet symmetric encryption.

Uses the ``cryptography`` library (BSD/Apache 2.0 license) to
provide AES-128-CBC encryption via the Fernet scheme.  A passphrase
is derived into a key using PBKDF2-HMAC-SHA256.
"""

import base64
import os
from pathlib import Path

from voxfusion.exceptions import VoxFusionError
from voxfusion.logging import get_logger

log = get_logger(__name__)

_SALT_SIZE = 16
_PBKDF2_ITERATIONS = 480_000


class EncryptionError(VoxFusionError):
    """Encryption or decryption failed."""


def derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a Fernet-compatible key from a passphrase using PBKDF2.

    Args:
        passphrase: User-provided passphrase.
        salt: Random salt bytes.

    Returns:
        URL-safe base64-encoded 32-byte key suitable for Fernet.
    """
    try:
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
    except ImportError as exc:
        raise EncryptionError(
            "cryptography package is not installed. "
            "Install with: pip install cryptography"
        ) from exc

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_PBKDF2_ITERATIONS,
    )
    key = kdf.derive(passphrase.encode("utf-8"))
    return base64.urlsafe_b64encode(key)


def encrypt_bytes(data: bytes, passphrase: str) -> bytes:
    """Encrypt data using Fernet with a passphrase-derived key.

    The output format is: ``salt (16 bytes) || fernet_token``.

    Args:
        data: Plaintext bytes to encrypt.
        passphrase: Passphrase to derive the encryption key.

    Returns:
        Encrypted bytes (salt + Fernet token).
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise EncryptionError(
            "cryptography package is not installed. "
            "Install with: pip install cryptography"
        ) from exc

    salt = os.urandom(_SALT_SIZE)
    key = derive_key(passphrase, salt)
    fernet = Fernet(key)
    token = fernet.encrypt(data)

    log.debug("encryption.encrypted", size=len(data))
    return salt + token


def decrypt_bytes(encrypted: bytes, passphrase: str) -> bytes:
    """Decrypt Fernet-encrypted data.

    Expects the format produced by ``encrypt_bytes``:
    ``salt (16 bytes) || fernet_token``.

    Args:
        encrypted: Encrypted bytes (salt + Fernet token).
        passphrase: Passphrase used during encryption.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        EncryptionError: If decryption fails (wrong passphrase, corrupted data).
    """
    try:
        from cryptography.fernet import Fernet, InvalidToken
    except ImportError as exc:
        raise EncryptionError(
            "cryptography package is not installed. "
            "Install with: pip install cryptography"
        ) from exc

    if len(encrypted) < _SALT_SIZE + 1:
        raise EncryptionError("Encrypted data is too short")

    salt = encrypted[:_SALT_SIZE]
    token = encrypted[_SALT_SIZE:]

    key = derive_key(passphrase, salt)
    fernet = Fernet(key)

    try:
        plaintext = fernet.decrypt(token)
    except InvalidToken as exc:
        raise EncryptionError(
            "Decryption failed. Wrong passphrase or corrupted data."
        ) from exc

    log.debug("encryption.decrypted", size=len(plaintext))
    return plaintext


def encrypt_file(
    input_path: str | Path,
    output_path: str | Path,
    passphrase: str,
) -> Path:
    """Encrypt a file and write the result to a new file.

    Args:
        input_path: Path to the plaintext file.
        output_path: Path for the encrypted output file.
        passphrase: Passphrase for encryption.

    Returns:
        Path to the encrypted file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    data = input_path.read_bytes()
    encrypted = encrypt_bytes(data, passphrase)
    output_path.write_bytes(encrypted)

    log.info(
        "encryption.file_encrypted",
        input=str(input_path),
        output=str(output_path),
    )
    return output_path


def decrypt_file(
    input_path: str | Path,
    output_path: str | Path,
    passphrase: str,
) -> Path:
    """Decrypt a file and write the result to a new file.

    Args:
        input_path: Path to the encrypted file.
        output_path: Path for the decrypted output file.
        passphrase: Passphrase used during encryption.

    Returns:
        Path to the decrypted file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    encrypted = input_path.read_bytes()
    plaintext = decrypt_bytes(encrypted, passphrase)
    output_path.write_bytes(plaintext)

    log.info(
        "encryption.file_decrypted",
        input=str(input_path),
        output=str(output_path),
    )
    return output_path
