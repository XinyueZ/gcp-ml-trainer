from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
import requests
import argparse
import json
from rich.pretty import pprint as pp
from Crypto.Cipher import AES
import scrypt, os, binascii

ENCODE = "utf-8"


def get_cipher_rsa():
    with open("private_key.pem", "r") as private_key_file:
        private_key_str = private_key_file.read()
        private_key = RSA.import_key(private_key_str)
        cipher_rsa = PKCS1_OAEP.new(private_key)
    return cipher_rsa


def rsa_decrypt(encrypted_data):
    """
    Decrypts the given encrypted data using RSA encryption.

    Args:
        encrypted_data (str): The encrypted data to be decrypted.

    Returns:
        bytes: The decrypted data.
    """
    encrypted_data = base64.b64decode(encrypted_data)
    cipher_rsa = get_cipher_rsa()
    decrypted_data = cipher_rsa.decrypt(encrypted_data)
    return decrypted_data

def aes_decrypt(encrypted_data, aes_key, iv, auth_tag):
    """
    Decrypts the given encrypted data using AES encryption.

    Args:
        encrypted_data (str): The encrypted data to be decrypted.
        aes_key (bytes): The AES key used for decryption.
        iv (bytes): The initialization vector.
        auth_tag (bytes): The authentication tag.
    Returns:
        bytes: The decrypted data.
    """
    encrypted_data = base64.b64decode(encrypted_data)
    iv = base64.b64decode(iv)
    auth_tag = base64.b64decode(auth_tag)
    cipher_aes = AES.new(aes_key, AES.MODE_GCM, iv)
    return cipher_aes.decrypt_and_verify(encrypted_data, auth_tag)


def main(args):
    local_url = "http://localhost:8080"
    message = args.message
    encrypting = args.encrypting
    params = {"message": message, "encrypting": encrypting}
    response = requests.get(local_url, params=params)

    if response.status_code != 200:
        pp(f"Error: {response.text}")
        return

    if encrypting == "false":
        res = json.dumps(response.text)
        pp(res)
        return
    pp(response.headers)
    aes_key = rsa_decrypt(response.headers["X-Encrypted-AES-Key"])
    encrypted_message = response.text
    decrypted_response = aes_decrypt(
        encrypted_message,
        aes_key,
        response.headers["X-AES-IV"],
        response.headers["X-AES-Auth-Tag"],
    )
    
    decrypted_response = json.dumps(decrypted_response.decode(ENCODE))
    pp(decrypted_response)


"""
To test the client.py script, run the following command:
python client.py --message "hello" --encrypting "false"
python client.py --message "hello" --encrypting "true"
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # two args: message and encrypting, both strings
    parser.add_argument(
        "--message",
        type=str,
        default="Hello, World!",
        help="The message to be encrypted.",
    )
    parser.add_argument(
        "--encrypting",
        type=str,
        default="true",
        help="Whether to encrypt the message or not.",
    )
    args = parser.parse_args()

    main(args)
