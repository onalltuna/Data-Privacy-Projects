import hashlib


def dictionaty_attack():

    dict_salted = {}
    dict_unhashed = {}
    passwords = []

    with open('salty-digitalcorp.txt', 'r') as file:
        lines = file.readlines()[1:] 
        for line in lines:
            username, salt, hash_outcome = line.strip().split(',')
            dict_salted[username] = [salt, hash_outcome]
    
    with open('rockyou.txt', 'r') as file:
        for line in file:
            passwords.append(line.strip())
    

    for password in passwords:
        for key,value in dict_salted.items():
            option1 = hasher(password + value[0])
            option2 = hasher(value[0] + password)
            if option1 == value[1] or option2 == value [1]:
                dict_unhashed[key] = password
    
    print(dict_unhashed)

def hasher(password):

    sha512_hash_obj = hashlib.sha512()
    sha512_hash_obj.update(password.encode('utf-8'))
    hashed_text = sha512_hash_obj.hexdigest()

    return hashed_text



    



dictionaty_attack()