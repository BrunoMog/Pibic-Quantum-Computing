import re

# Lista original
lista_original = ['x0' 'x1' 'x2' 'x3' 'x4' 'x5' 'x6' 'x7' 'x8' 'x9' 'x10' 'x11' 'x12' 'x13'
 'x15' 'x16' 'x17' 'x18' 'x19' 'x20' 'x21']

# Extrai os números usando expressão regular
numeros = [int(re.sub(r'[^\d]', '', item)) for item in lista_original]

print(numeros)