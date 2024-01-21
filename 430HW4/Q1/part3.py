import hashlib

def dictionary_attack():
    dict_hashed = {}
    dict_unhashed = {}
    passwords = []

    with open('keystreching-digitalcorp.txt', 'r') as file:
        lines = file.readlines()[1:] 
        for line in lines:
            username, salt, hashed_password = line.strip().split(',')
            dict_hashed[username] = (salt, hashed_password)

    with open('rockyou.txt', 'r') as file:
        for line in file:
            passwords.append(line.strip())

#xi + password + salt
    for password in passwords:
        for key, value in dict_hashed.items():
            first_hashed = hasher(password + value[0])
            option = hasher(first_hashed + password + value[0])
            if option == value[1]:
                    dict_unhashed[key] = password
                    # del dict_hashed[key]
                    break  
            else: 
                old_option = option
                for i in range(2000):
                    new_option =  hasher(old_option + password + value[0])
                    if new_option == value[1]:
                        dict_unhashed[key] = password
                        # del dict_hashed[key]
                        break  
                    else:
                        old_option = new_option
# salt + xi + password
    for password in passwords:
        for key, value in dict_hashed.items():
            first_hashed = hasher(value[0] + password)
            option = hasher(value[0] + first_hashed + password)
            if option == value[1]:
                    dict_unhashed[key] = password
                    # del dict_hashed[key]
                    break  
            else: 
                old_option = option
                for i in range(2000):
                    new_option =  hasher(value[0] + old_option + password)
                    if new_option == value[1]:
                        dict_unhashed[key] = password
                        # del dict_hashed[key]
                        break  
                    else:
                        old_option = new_option
# password + salt + xi
    for password in passwords:
        for key, value in dict_hashed.items():
            first_hashed = hasher(password + value[0])
            option = hasher(password + value[0] + first_hashed)
            if option == value[1]:
                    dict_unhashed[key] = password
                    # del dict_hashed[key]
                    break  
            else: 
                old_option = option
                for i in range(2000):
                    new_option =  hasher(password + value[0] + old_option)
                    if new_option == value[1]:
                        dict_unhashed[key] = password
                        # del dict_hashed[key]
                        break  
                    else:
                        old_option = new_option
#xi + salt + password
    for password in passwords:
        for key, value in dict_hashed.items():
            first_hashed = hasher(value[0] + password)
            option = hasher(first_hashed + value[0] + password)
            if option == value[1]:
                    dict_unhashed[key] = password
                    # del dict_hashed[key]
                    break  
            else: 
                old_option = option
                for i in range(2000):
                    new_option =  hasher(old_option + value[0] + password)
                    if new_option == value[1]:
                        dict_unhashed[key] = password
                        # del dict_hashed[key]
                        break  
                    else:
                        old_option = new_option
# password + xi + salt
    for password in passwords:
        for key, value in dict_hashed.items():
            first_hashed = hasher(password + value[0])
            option = hasher(password + first_hashed + value[0])
            if option == value[1]:
                    dict_unhashed[key] = password
                    # del dict_hashed[key]
                    break  
            else: 
                old_option = option
                for i in range(2000):
                    new_option =  hasher(password + old_option + value[0])
                    if new_option == value[1]:
                        dict_unhashed[key] = password
                        # del dict_hashed[key]
                        break  
                    else:
                        old_option = new_option

# salt + password + xi




    print(dict_unhashed)

def hasher(password):

    sha512_hash_obj = hashlib.sha512()
    sha512_hash_obj.update(password.encode('utf-8'))
    hashed_text = sha512_hash_obj.hexdigest()

    return hashed_text

dictionary_attack()