#Import Library
import cv2
import numpy as np
from skimage import data
import matplotlib.pyplot as plt

img = cv2.imread('PicsArt_08-31-11.13.46.jpg', 0) # Membaca gambar dengan menggunakan OpenCV dan mengkonversinya menjadi citra grayscale
row, column = img.shape # Mendapatkan ukuran baris dan kolom citra
img1 = np.zeros((row,column),dtype = 'uint8') # Membuat array kosong dengan ukuran yang sama dengan citra untuk menyimpan hasil akhir

min_range = 10 # Menentukan rentang nilai piksel minimum
max_range = 60 # Menentukan rentang nilai piksel maksimum

for i in range(row): # Melakukan perulangan sebanyak jumlah baris citra
    for j in range(column): # Melakukan perulangan sebanyak jumlah kolom citra
        if img[i,j]>min_range and img[i,j]<max_range: # Memeriksa apakah nilai piksel berada dalam rentang yang ditentukan
            img1[i,j] = 255 # Jika iya, atur piksel ke putih (255)
else:
    img1[i,j] = 0 # Jika tidak, atur piksel ke hitam (0)

#Plot Image
fig, axes = plt.subplots(2, 2, figsize=(12, 12)) # Membuat subplot dengan ukuran 2 baris dan 2 kolom untuk menampilkan gambar dan histogram
ax = axes.ravel() # Melakukan unravel pada objek axes

ax[0].imshow(img, cmap=plt.cm.gray) # Menampilkan citra asli
ax[0].set_title("Citra Input") # Memberikan judul pada citra asli
ax[1].hist(img.ravel(), bins=256) # Menampilkan histogram citra asli
ax[1].set_title('Histogram Input') # Memberikan judul pada histogram citra asli

ax[2].imshow(img1, cmap=plt.cm.gray) # Menampilkan citra hasil thresholding
ax[2].set_title("Citra Output") # Memberikan judul pada citra hasil thresholding
ax[3].hist(img1.ravel(), bins=256) # Menampilkan histogram citra hasil thresholding
ax[3].set_title('Histogram Output') # Memberikan judul pada histogram citra hasil thresholding

plt.show() # Menampilkan plot citra dan histogram