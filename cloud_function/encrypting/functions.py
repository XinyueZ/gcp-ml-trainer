from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import json
import base64

ENCODE = "utf-8"


def get_cipher_rsa():
    with open("public_key.pem", "r") as public_key_file:
        public_key_str = public_key_file.read()
        public_key = RSA.import_key(public_key_str)
        cipher_rsa = PKCS1_OAEP.new(public_key)
    return cipher_rsa


def encrypt_data(data):
    response_json = json.dumps(data).encode(ENCODE)
    cipher_rsa = get_cipher_rsa()
    encrypted_data = cipher_rsa.encrypt(response_json)
    return base64.b64encode(encrypted_data).decode(ENCODE)


def hello_encrypting(params):
    """
    Encrypts the given message if 'encrypting' parameter is set to 'true'.

    call:
        Not encrypt the message:
        curl http://localhost:8080\?message\=Hello\&\encrypting\=false

        Encrypt the message:
        curl http://localhost:8080\?message\=Hello\&\encrypting\=true
        curl http://localhost:8080\?message\=Hello
    Args:
        params (dict): A dictionary containing the function parameters.

    Returns:
        str: The encrypted data if 'encrypting' is set to 'true', otherwise a JSON string containing the response data.

    """
    message = params.args.get("message", "Hello, World!")
    is_encrypting = params.args.get("encrypting", "true").lower() == "true"
    response_data = {
        "code": 200,
        "message": message,
        "encrypted": is_encrypting,
    }
    if not is_encrypting:
        return json.dumps(response_data)
    encrypted_data = encrypt_data(response_data)
    return encrypted_data
