import hashlib


def dictionaty_attack():

    dict_hashed = {}
    dict_unhashed = {}
    passwords = []

    with open('digitalcorp.txt', 'r') as file:
        lines = file.readlines()[1:] 
        for line in lines:
            username, hashed_password = line.strip().split(',')
            dict_hashed[username] = hashed_password
    
    with open('rockyou.txt', 'r') as file:
        for line in file:
            passwords.append(line.strip())
    

    for password in passwords:
        hashed_password = hasher(password)
        for key,value in dict_hashed.items():
            if value == hashed_password:
                dict_unhashed[key] = password
    
    print(dict_unhashed)

def hasher(password):

    sha512_hash_obj = hashlib.sha512()
    sha512_hash_obj.update(password.encode('utf-8'))
    hashed_text = sha512_hash_obj.hexdigest()

    return hashed_text



    



dictionaty_attack()