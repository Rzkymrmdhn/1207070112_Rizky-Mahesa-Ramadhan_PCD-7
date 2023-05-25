#Import Library
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
from skimage import data

#Read Image
image = cv2.imread('PicsArt_08-31-11.13.46.jpg', 0) # Membaca gambar dengan menggunakan OpenCV dan mengkonversinya menjadi citra grayscale

#Penerapan Histogram Equalization (HE)
image_equalized = cv2.equalizeHist(image)

#Penerapan Metode Contrast Limited Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)) # Membuat objek CLAHE dengan batasan clipLimit=2 dan ukuran tileGridSize=(8, 8)

#Apply CLAHE to the original image
image_clahe = clahe.apply(image) # Membuat objek CLAHE dengan batasan clipLimit=2 dan ukuran tileGridSize=(8, 8)

#Penerapan metode Contrast Stretching (CS)
# Create an empty array to store the final output
image_cs = np.zeros((image.shape[0],image.shape[1]),dtype = 'uint8') # Membuat array kosong dengan ukuran yang sama dengan citra asli untuk menyimpan hasil akhir

# Apply Min-Max Contrasting
min = np.min(image) # Mencari nilai piksel minimum pada citra
max = np.max(image) # Mencari nilai piksel maksimum pada citra

for i in range(image.shape[0]): # Melakukan perulangan sebanyak jumlah baris citra
    for j in range(image.shape[1]): # Melakukan perulangan sebanyak jumlah kolom citra
        image_cs[i,j] = 255*(image[i,j]-min)/(max-min) # Melakukan perulangan sebanyak jumlah kolom citra

#Penerapan Metode Perkalian Konstanta
copyCamera = image.copy().astype(float) # Membuat salinan citra dan mengkonversinya menjadi tipe data float

m1,n1 = copyCamera.shape # Membuat salinan citra dan mengkonversinya menjadi tipe data float
output1 = np.empty([m1, n1]) # Membuat array kosong dengan ukuran yang sama dengan citra salinan

for baris in range(0, m1-1): # Melakukan perulangan sebanyak jumlah baris citra salinan
    for kolom in range(0, n1-1): # Melakukan perulangan sebanyak jumlah kolom citra salinan
        a1 = baris
        b1 = kolom
        output1[a1, b1] = copyCamera[baris, kolom] * 1.9 # Mengaplikasikan metode perkalian konstanta pada setiap piksel citra salinan

#Plot Image
fig, axes = plt.subplots(5, 2, figsize=(20, 20)) # Membuat subplot dengan ukuran 5 baris dan 2 kolom untuk menampilkan gambar dan histogram
ax = axes.ravel() # Melakukan unravel pada objek axes

ax[0].imshow(image, cmap=plt.cm.gray) # Menampilkan citra asli
ax[0].set_title("Citra Input") # Memberikan judul pada citra asli
ax[1].hist(image.ravel(), bins=256) # Menampilkan histogram citra asli
ax[1].set_title('Histogram Input') # Memberikan judul pada histogram citra asli

ax[2].imshow(image_equalized, cmap=plt.cm.gray) # Menampilkan citra hasil Histogram Equalization (HE)
ax[2].set_title("Citra Output HE") # Memberikan judul pada citra hasil Histogram Equalization (HE)
ax[3].hist(image_equalized.ravel(), bins=256) # Menampilkan histogram citra hasil Histogram Equalization (HE)
ax[3].set_title('Histogram Output HE Method') # Memberikan judul pada histogram citra hasil Histogram Equalization (HE)

ax[4].imshow(image_cs, cmap=plt.cm.gray) # Menampilkan citra hasil Contrast Stretching (CS)
ax[4].set_title("Citra Output CS") # Memberikan judul pada citra hasil Contrast Stretching (CS)
ax[5].hist(image_cs.ravel(), bins=256) # Menampilkan histogram citra hasil Contrast Stretching (CS)
ax[5].set_title('Histogram Output CS Method') # Memberikan judul pada histogram citra hasil Contrast Stretching (CS)

ax[6].imshow(image_clahe, cmap=plt.cm.gray) # Menampilkan citra hasil Contrast Limited Adaptive Histogram Equalization (CLAHE)
ax[6].set_title("Citra Grayscale CLAHE") # Memberikan judul pada citra hasil Contrast Limited Adaptive Histogram Equalization (CLAHE)
ax[7].hist(image_clahe.ravel(), bins=256) # Menampilkan histogram citra hasil Contrast Limited Adaptive Histogram Equalization (CLAHE)
ax[7].set_title('Histogram Output CLAHE Method') # Memberikan judul pada histogram citra hasil Contrast Limited Adaptive Histogram Equalization (CLAHE)

ax[8].imshow(output1, cmap=plt.cm.gray) # Menampilkan citra hasil perkalian konstanta
ax[8].set_title("Citra Grayscale Perkalian Konstanta") # Memberikan judul pada citra hasil perkalian konstanta
ax[9].hist(output1.ravel(), bins=256) # Menampilkan histogram citra hasil perkalian konstanta
ax[9].set_title('Histogram Output Perkalian Konstanta Method') # Memberikan judul pada histogram citra hasil perkalian konstanta

fig.tight_layout() # Mengatur tata letak subplot agar terlihat rapi
plt.show() # Menampilkan plot citra dan histogram