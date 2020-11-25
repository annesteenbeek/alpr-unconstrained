import itertools
import re
import sys

number_regex = re.compile(r'[1-9][0-9]{3}')
weights = (9, 4, 5, 4, 3, 2)
VALID_CHECKSUMS = tuple('A, Z, Y, X, U, T, S, R, P, M, L, K, J, H, G, E, D, C, B'.split(', '))
denominion = len(VALID_CHECKSUMS)


if sys.version_info[0] < 3:
    def is_string(string):
        return isinstance(string, basestring)
else:
    def is_string(string):
        return isinstance(string, str)


def is_valid(license_number):
    if not is_string(license_number):
        return False
    if len(license_number) != 8:
        return False
    elif license_number.islower():
        return False
    elif 'I' in license_number or 'O' in license_number:
        return False
    elif license_number[:3] not in VALID_HEADERS:
        return False
    elif number_regex.match(license_number[3:7]) is None:
        return False
    elif license_number[7] not in VALID_CHECKSUMS:
        return False
    else:
        return get_checksum(license_number[1:-1]) == license_number[7]


VALID_HEADERS = ['S__'] + ['GB_']
consonants = 'BCDFGHJKLMNPQRSTVWXYZ'
vows = 'AEIOU'
VALID_HEADERS = [x+y+z for x,y,z in itertools.product(['S'], consonants, consonants + vows)]
VALID_HEADERS += [x+y for x,y in itertools.product(['GB', 'GA', 'GE', 'GU'], consonants+vows)]


def get_checksum(string):
    alphabet = string[:2]
    numbers = string[2:]
    code = list(ord(x) - 64 for x in alphabet)
    code += list(map(int, numbers))
    check = sum(x * y for x,y in zip(code, weights))
    return VALID_CHECKSUMS[check % denominion]


if __name__ == '__main__':
    import json
    plates = json.load(open('valide_annotations.json'))
    vocab = '0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ'
    prior = [[0] * 8 for _ in range(len(vocab))]
    # getting prior from data
    for p in plates:
        if is_valid(p):
            for i, c in enumerate(p):
                prior[vocab.index(c)][i] += 1
    file = open('prior.csv', 'w')
    for l in prior:
        file.write(','.join(map(str, l)) + '\n')
    file.close()
    # print(get_checksum('KV6201') == 'B')  # SKV6201B
