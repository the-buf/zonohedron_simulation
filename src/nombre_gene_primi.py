# Make sure that you have all these libaries available to run the code successfully
import os
import math
import time
import csv


# Save data to this file
file_to_save_1 = "/home/theo/Documents/coding/these/nb_gene_primitifs.csv"
file_to_save_2 = "/home/theo/Documents/coding/these/nb_gene_primitifs2D.csv"

t1 = time.perf_counter()


def nb_generateur_primitifs(n):
    result = 0
    if n == 1:
        result = 3
    if n >= 2:
        for i in range(1, n):
            if math.gcd(i, n) == 1:
                result += 6
            if i < n - 1:
                for j in range(i + 1, n):
                    if math.gcd(i, math.gcd(j - i, n - j)) == 1:
                        result += 4
    return result


def phi(n):
    # calcule l'indicatrice d'Euler
    result = 1
    if n > 2:
        for i in range(2, n):
            if math.gcd(i, n) == 1:
                result += 1
    return result


if not os.path.exists(file_to_save_1):
    with open(file_to_save_1, "w") as myfile_1:
        with open(file_to_save_2, "w") as myfile_2:
            i = 0
            while time.perf_counter() - t1 < 2500:
                i += 1
                if i % 500 == 0:
                    print(i, " en ", time.perf_counter - t1, " secondes")
                writer_1 = csv.writer(myfile_1, quoting=csv.QUOTE_ALL)
                writer_1.writerow([nb_generateur_primitifs(i)])
                writer_2 = csv.writer(myfile_2, quoting=csv.QUOTE_ALL)
                writer_2.writerow([3 * phi(i) if (i == 1) else 6 * phi(i)])


# If the data is already there, just load it from the CSV
else:
    print("File already exists. Loading data from CSV")
