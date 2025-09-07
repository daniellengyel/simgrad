import requests
from bs4 import BeautifulSoup

import numpy as np

def convert_pm(had):
    # Split into lines and remove empty ones
    lines_raw = [line for line in had.strip().split('\n')]
    lines = []
    for l in lines_raw:
        if l[0] == "+" or l[0] == "-":
            lines.append(l)
    
    
    # Convert each character: '+' → 1, '-' → -1
    matrix = [[1 if char == '+' else -1 for char in line] for line in lines]
    return np.array(matrix) 

def is_orthogonal(matrix, tol=1e-8):
    matrix = np.array(matrix)
    identity = np.eye(matrix.shape[0])
    return np.allclose(matrix.T @ matrix, identity, atol=tol)

# ===== Pull index and infer links ======
print("Getting index and links.")

url = 'http://neilsloane.com/hadamard/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract all links
links = [a['href'] for a in soup.find_all('a', href=True)]

# Get had links
had_links = []
for l in links:
    if 'had' in l:
        had_links.append(l)

# ===== Pull initial Had matrices ======
print("Pulling Initial Hadamard Matrices.")
hads = {}

for l in had_links:
    if len(l.split(".")) > 4:
        continue

    n = l.split(".")[1]
    if n in hads:
        continue

    response = requests.get(url + l)
    hads[n] = response.text


# ===== Convert and Confirm Had Matrices =====
print("Converting Hadamard Matrices.")
had_mats = {}
for n in hads:
    had = hads[n]

    if had[0] != "+" and had[0] != '-':
        print(n)
        continue

    had_mat_curr = convert_pm(had)

    # make first row all ones 
    had_mat_curr = had_mat_curr * had_mat_curr[0]
    had_mats[n] = had_mat_curr / np.sqrt(int(n))
    
print("Asserting Had matrices are orthogonal.")
for n in had_mats:
    mat = had_mats[n]
    print(n)
    assert is_orthogonal(mat)

# ===== Save Data =====
print("Saving data.")
import os 
import pickle

dest = os.getcwd() + "/sample_set/CASG/hadamard/data"
for n in had_mats:
    with open(f'{dest}/had_{n}.pkl', 'wb') as f:
        pickle.dump(had_mats[n], f)