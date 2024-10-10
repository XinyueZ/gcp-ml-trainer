import base64
import argparse

ENCODE = "utf-8"


def file_to_base64(filefullpath: str) -> str:
    with open(filefullpath, "rb") as file:
        file_content = file.read()
        return base64.b64encode(file_content).decode(ENCODE)


# Example:
# python file_to_base64.py /path/to/file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert file to base64.")
    parser.add_argument(
        "filefullpath", type=str, help="The full path to a file to convert to base64."
    )
    args = parser.parse_args()
    print(file_to_base64(args.filefullpath))
