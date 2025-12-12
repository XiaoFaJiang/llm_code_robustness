import base64
import binascii
import json
import re


import_success = True
try:
    from Crypto.Cipher import DES
    from Crypto.Util.Padding import pad, unpad
except:
    import_success = False

ENCRYPTED_PATTERN = re.compile(r"!!!(.*?)!!!ENCRYPTED_A(\d+)V(\d+)<<<(.*?)>>>")
ENCRYPTED_TEMPLATE = "!!!机密信息，越权解密将被追责!!!ENCRYPTED_A%sV%s<<<%s>>>"
# 当前加密算法版本
CURRENT_ALGORITHM_VERISON = 1
CURRENT_KEY_VERSION = 1
# 假设从KMS获取的密钥版本映射
KEY_VERSION_MAP = json.loads('{"1": "0ba123cc75f205e1"}')

def auto_decrypt(encrypted_text):
    if not import_success:
        print("WARNING: 当前镜像未安装数据加密相关依赖，因此跳过自动解密解密直接返回文本")
        return encrypted_text
    if not encrypted_text:
        return ""
    decrypted_text = encrypted_text
    matcher_start = 0
    while True:
        match = ENCRYPTED_PATTERN.search(decrypted_text, matcher_start)
        if not match:
            break
        match_full_text = match.group()
        algorithm_version = int(match.group(2))
        key_version = int(match.group(3))
        encrypted_content = match.group(4)
        key = KEY_VERSION_MAP.get(str(key_version))
        if not key:
            raise Exception(f"autoDecrypt unsupported keyVersion[{key_version}]")
        if algorithm_version == 1:
            decrypted_content = decrypt_content_with_des_base64(encrypted_content, key)
        else:
            raise Exception(f"autoDecrypt unsupported algorithmVersion[{algorithm_version}]")
        decrypted_text = decrypted_text.replace(match_full_text, decrypted_content)
        matcher_start = match.end()
    return decrypted_text

def decrypt_content_with_des_base64(encrypted_content, hex_key):
    key_bytes = binascii.unhexlify(hex_key)
    encrypted_bytes = base64.b64decode(encrypted_content)
    cipher = DES.new(key_bytes, DES.MODE_ECB)
    decrypted_bytes = unpad(cipher.decrypt(encrypted_bytes), DES.block_size)
    return decrypted_bytes.decode()

def auto_encrypt(decrypted_content):
    if not import_success:
        raise Exception("当前镜像未安装数据加密相关依赖，无法使用保密数据集")
    key = KEY_VERSION_MAP.get(str(CURRENT_KEY_VERSION))
    if not key:
        raise Exception(f"autoEncrypt unsupported keyVersion[{CURRENT_KEY_VERSION}]")
    if CURRENT_ALGORITHM_VERISON == 1:
        encrypted_content = encrypt_content_with_des_base64(decrypted_content, key)
    else:
        raise Exception(f"autoEncrypt unsupported algorithmVersion[{CURRENT_ALGORITHM_VERISON}]")
    return ENCRYPTED_TEMPLATE % (CURRENT_ALGORITHM_VERISON, CURRENT_KEY_VERSION, encrypted_content)

def encrypt_text(content, encrypt):
    if encrypt:
        return auto_encrypt(content)
    return content

def safe_print(text, encrypt=False):
    target = text if isinstance(text, str) else str(text)
    target = encrypt_text(target, encrypt)
    print(target)

def encrypt_content_with_des_base64(decrypted_content, hex_key):
    decrypted_bytes = pad(decrypted_content.encode(), DES.block_size)
    key_bytes = binascii.unhexlify(hex_key)
    cipher = DES.new(key_bytes, DES.MODE_ECB)
    encrypted_bytes = cipher.encrypt(decrypted_bytes)
    encrypted_content = base64.b64encode(encrypted_bytes).decode()
    return encrypted_content

def hex_string_to_byte_array(s):
    return bytes.fromhex(s)


def load_and_decrypt_raw_result(path):
    return json.loads(auto_decrypt(open(path, "r",encoding="utf8").read()))["request_states"]

def upload_file(file):
    pass

if __name__ == "__main__":
    plain_text = """hello,美团.hello,world"""
    encrypted_text = auto_encrypt(plain_text)
    encrypted_text = encrypted_text + "\n lalalala \n" + auto_encrypt(plain_text)
    print(f"en: {encrypted_text}")
    decrypted_text = auto_decrypt(encrypted_text)
    print(f"de: {decrypted_text}")