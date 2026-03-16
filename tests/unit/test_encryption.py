"""Tests for the encryption module."""

import pytest

from voxfusion.security.encryption import (
    EncryptionError,
    decrypt_bytes,
    decrypt_file,
    derive_key,
    encrypt_bytes,
    encrypt_file,
)


class TestDeriveKey:
    """Tests for key derivation."""

    def test_produces_bytes(self):
        key = derive_key("password", b"0123456789abcdef")
        assert isinstance(key, bytes)

    def test_deterministic(self):
        salt = b"fixed_salt_value"
        key1 = derive_key("password", salt)
        key2 = derive_key("password", salt)
        assert key1 == key2

    def test_different_passwords_different_keys(self):
        salt = b"fixed_salt_value"
        key1 = derive_key("password1", salt)
        key2 = derive_key("password2", salt)
        assert key1 != key2

    def test_different_salts_different_keys(self):
        key1 = derive_key("password", b"salt_aaaaaaaaaaaaa")
        key2 = derive_key("password", b"salt_bbbbbbbbbbbbb")
        assert key1 != key2


class TestEncryptDecrypt:
    """Tests for encrypt_bytes and decrypt_bytes."""

    def test_roundtrip(self):
        data = b"Hello, world! This is secret."
        passphrase = "my-secret-passphrase"
        encrypted = encrypt_bytes(data, passphrase)
        decrypted = decrypt_bytes(encrypted, passphrase)
        assert decrypted == data

    def test_encrypted_differs_from_plaintext(self):
        data = b"plaintext data"
        encrypted = encrypt_bytes(data, "pass")
        assert encrypted != data

    def test_wrong_passphrase_fails(self):
        data = b"secret data"
        encrypted = encrypt_bytes(data, "correct-password")
        with pytest.raises(EncryptionError, match="Decryption failed"):
            decrypt_bytes(encrypted, "wrong-password")

    def test_corrupted_data_fails(self):
        data = b"secret data"
        encrypted = encrypt_bytes(data, "pass")
        corrupted = encrypted[:16] + b"x" * (len(encrypted) - 16)
        with pytest.raises(EncryptionError):
            decrypt_bytes(corrupted, "pass")

    def test_too_short_data_fails(self):
        with pytest.raises(EncryptionError, match="too short"):
            decrypt_bytes(b"short", "pass")

    def test_empty_data_roundtrip(self):
        data = b""
        encrypted = encrypt_bytes(data, "pass")
        decrypted = decrypt_bytes(encrypted, "pass")
        assert decrypted == data

    def test_large_data_roundtrip(self):
        data = b"A" * 1_000_000
        encrypted = encrypt_bytes(data, "pass")
        decrypted = decrypt_bytes(encrypted, "pass")
        assert decrypted == data


class TestFileEncryption:
    """Tests for file encryption/decryption."""

    def test_file_roundtrip(self, tmp_path):
        input_file = tmp_path / "secret.txt"
        encrypted_file = tmp_path / "secret.enc"
        decrypted_file = tmp_path / "secret.dec.txt"

        input_file.write_text("Top secret content!")
        encrypt_file(input_file, encrypted_file, "password123")

        assert encrypted_file.exists()
        assert encrypted_file.read_bytes() != input_file.read_bytes()

        decrypt_file(encrypted_file, decrypted_file, "password123")
        assert decrypted_file.read_text() == "Top secret content!"

    def test_file_wrong_passphrase(self, tmp_path):
        input_file = tmp_path / "data.txt"
        encrypted_file = tmp_path / "data.enc"
        output_file = tmp_path / "data.dec"

        input_file.write_text("data")
        encrypt_file(input_file, encrypted_file, "pass1")

        with pytest.raises(EncryptionError):
            decrypt_file(encrypted_file, output_file, "pass2")
