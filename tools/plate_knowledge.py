weights = (9, 4, 5, 4, 3, 2)
symbols = tuple('A, Z, Y, X, U, T, S, R, P, M, L, K, J, H, G, E, D, C, B'.split(', '))
denominion = len(symbols)


def get_checksum(string):
    assert len(string) == 6
    alphabet = string[:2].upper()
    assert alphabet.isalpha()
    numbers = string[2:]
    code = list(ord(x) - 64 for x in alphabet)
    code += list(map(int, numbers))
    check = sum(x * y for x,y in zip(code, weights))
    return symbols[check % denominion]


prefix = 


if __name__ == '__main__':
    print(get_checksum('KV6201') == 'B')  # SKV6201B
