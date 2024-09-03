import functions_framework
import pandas as pd
from google.cloud import bigquery
from json import loads, dumps
from loguru import logger
import os
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
import json
import base64
import hashlib
from Crypto.Random import get_random_bytes
import scrypt

ENCODE = "utf-8"
LENGTH = 16
SALT = os.urandom(LENGTH)
AES_KEY = scrypt.hash(
    get_random_bytes(LENGTH).hex(),
    SALT,
    N=16384,
    r=8,
    p=1,
    buflen=32,
)


def get_cipher_rsa():
    with open("public_key.pem", "r") as public_key_file:
        public_key_str = public_key_file.read()
        public_key = RSA.import_key(public_key_str)
        cipher_rsa = PKCS1_OAEP.new(public_key)
    return cipher_rsa


def get_cipher_aes(cipher_rsa=None):
    cipher_aes = AES.new(AES_KEY, AES.MODE_GCM, nonce=bytes(12))
    if cipher_rsa:
        encrypted_aes_key = cipher_rsa.encrypt(AES_KEY)
        encrypted_aes_key = base64.b64encode(encrypted_aes_key).decode(ENCODE)
    return cipher_aes, encrypted_aes_key if cipher_rsa else None


def encrypt_data(data_in_bytes):
    cipher_rsa = get_cipher_rsa()
    cipher_aes, encrypted_aes_key = get_cipher_aes(cipher_rsa)
    ciphertext, auth_tag = cipher_aes.encrypt_and_digest(data_in_bytes)
    encrypted_data = base64.b64encode(ciphertext).decode(ENCODE)
    return encrypted_data, cipher_aes.nonce, auth_tag, encrypted_aes_key


@functions_framework.http
def hello_encrypting(params):
    """
    Encrypts the given message if 'encrypting' parameter is set to 'true'.

    call:
        Not encrypt the message:
        curl http://localhost:8080\?message\=Hello\&\encrypting\=false

        Encrypt the message:
        curl http://localhost:8080\?message\=Hello\&\encrypting\=true
        curl http://localhost:8080\?message\=Hello

    deploy:
        local:
            functions-framework --target hello_encrypting --source main.py
        gcp:
            gcloud functions deploy hello_encrypting \
                    --gen2 \
                    --runtime python312 \
                    --trigger-http \
                    --allow-unauthenticated \
                    --entry-point "hello_encrypting" \
                    --region "europe-west1" \
                    --source .
    Args:
        params (dict): A dictionary containing the function parameters.

    Returns:
        str: The encrypted data if 'encrypting' is set to 'true', otherwise a JSON string containing the response data.

    """
    message = params.args.get("message", "Hello, World!")
    is_encrypting = params.args.get("encrypting", "true").lower() == "true"
    response_data = {
        "code": 666,
        "message": message,
        "encrypted": is_encrypting,
    }
    if not is_encrypting:
        return json.dumps(response_data)

    json_data = json.dumps(response_data).encode(ENCODE)

    encrypted_data, aes_iv, auth_tag, encrypted_aes_key = encrypt_data(json_data)
    headers = {
        "X-AES-SALT": base64.b64encode(SALT).decode(ENCODE),
        "X-AES-IV": base64.b64encode(aes_iv).decode(ENCODE),
        "X-AES-Auth-Tag": base64.b64encode(auth_tag).decode(ENCODE),
        "X-Encrypted-AES-Key": encrypted_aes_key,
    }
    return (encrypted_data, 200, headers)
