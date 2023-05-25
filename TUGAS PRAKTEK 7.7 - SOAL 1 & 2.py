# 1. LOW-PASS FILTERING
# Mengimpor library 
import cv2  
import numpy as np  
from matplotlib import pyplot as plt  

# Membaca citra dengan nama file 'zee.jpeg' menggunakan OpenCV
img = cv2.imread('jisoo.jpg')  

# Mengubah format citra dari BGR (default OpenCV) ke RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

# Membuat filter: matriks berukuran 5 x 5 dengan nilai setiap elemen adalah 1/25
kernel = np.ones((3, 3), np.float32) / 25

# Melakukan filtering
jisoo_filter = cv2.filter2D(img, -1, kernel)  # Melakukan filtering pada citra menggunakan filter kernel

# Membuat kernel berukuran 3 x 3 dengan setiap elemennya bernilai 1/25. Digunakan untuk filtering rata-rata.
kernel = np.ones((3, 3), np.float32) / 25

# Melakukan filtering pada citra 'img' menggunakan kernel 'kernel'. Hasil filtering disimpan dalam 'zee_filter'.
jisoo_filter = cv2.filter2D(img, -1, kernel)

# Mengatur ukuran tampilan plot menjadi 10x10 inch.
plt.rcParams["figure.figsize"] = (10, 10)
# Membuat subplot dengan 1 baris, 2 kolom, dan mengambil posisi pertama (1).
plt.subplot(121)

# Menampilkan citra asli.
plt.imshow(img)

# Memberikan judul pada subplot.
plt.title('Original')

# Menghilangkan label sumbu x dan y.
plt.xticks([])
plt.yticks([])

# Membuat subplot dengan 1 baris, 2 kolom, dan mengambil posisi kedua (2).
plt.subplot(122)
# Menampilkan hasil filtering.
plt.imshow(jisoo_filter)
# Memberikan judul pada subplot.
plt.title('Averaging')

# Menghilangkan label sumbu x dan y.
plt.xticks([])
plt.yticks([])
plt.show()

# Melakukan proses blurring pada citra 'img' menggunakan filter berukuran 5x5.
jisoo_blur = cv2.blur(img, (5, 5))

# Mengatur ukuran tampilan plot menjadi 10x5 inch.
plt.rcParams["figure.figsize"] = (10, 5)
# Membuat subplot dengan 2 baris, 2 kolom, dan mengambil posisi pertama (1).
plt.subplot(221)

# Menampilkan citra asli dengan menggunakan colormap 'gray'.
plt.imshow(img, cmap='gray')

# Memberikan judul pada subplot.
plt.title('Original')

# Menghilangkan label sumbu x dan y.
plt.xticks([])
plt.yticks([])

# Membuat subplot dengan 2 baris, 2 kolom, dan mengambil posisi kedua (2).
plt.subplot(222)

# Menampilkan citra hasil blurring dengan menggunakan colormap 'gray'.
plt.imshow(jisoo_blur, cmap='gray')

# Memberikan judul pada subplot.
plt.title('Gambar Blur')
# Menghilangkan label sumbu x dan y.
plt.xticks([])
plt.yticks([])

# Membuat subplot dengan 2 baris, 2 kolom, dan mengambil posisi ketiga (3).
plt.subplot(223)

# Menampilkan histogram dari citra asli.
plt.hist(img.ravel(), 256, [0, 256], color='blue', alpha=0.7)

# Memberikan judul pada subplot.
plt.title('Histogram Citra Asli')

# Memberikan label sumbu x dan y.
plt.xlabel('Intensitas Piksel')
plt.ylabel('Frekuensi')

# Membuat subplot dengan 2 baris, 2 kolom, dan mengambil posisi keempat (4).
plt.subplot(224)

# Menampilkan histogram dari citra hasil blurring.
plt.hist(jisoo_blur.ravel(), 256, [0, 256], color='green', alpha=0.7)

# Memberikan judul pada subplot.
plt.title('Histogram Citra Setelah Blur')

# Memberikan label sumbu x dan y.
plt.xlabel('Intensitas Piksel')
plt.ylabel('Frekuensi')

# Menyusun tata letak subplot secara otomatis.
plt.tight_layout()

# Menampilkan plot.
plt.show()

# ini adalah cara lain untuk membuat sebuah kernel, 
# yaitu dengan menggunakan np.matrix
# kali ini, ukuran matriksnya 3 x 3
# Membuat kernel
kernel = np.matrix([
          [1, 1, 1],
          [1, 2, 1],
          [1, 1, 1]         
          ])/25
# Membuat kernel dengan menggunakan np.matrix. Kernel ini berukuran 3x3 dan memiliki nilai tertentu yang digunakan untuk proses filtering.

# Melakukan filtering
jisoo_filter = cv2.filter2D(img, -1, kernel)

# Membuat plot
fig, axs = plt.subplots(2, 2, figsize=(10, 5))


# Plot gambar asli
# Menampilkan citra asli pada subplot pertama. Memberikan judul 'Original' pada subplot. Menghilangkan sumbu x dan y pada subplot.
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 0].axis('off')


# Plot hasil filtering
axs[0, 1].imshow(jisoo_filter, cmap='gray')
axs[0, 1].set_title('Averaging')
axs[0, 1].axis('off')


# Plot histogram citra asli
axs[1, 0].hist(img.ravel(), 256, [0, 256], color='blue', alpha=0.7)
axs[1, 0].set_title('Histogram Citra Asli')
axs[1, 0].set_xlabel('Intensitas Piksel')
axs[1, 0].set_ylabel('Frekuensi')

# Plot histogram citra setelah filtering
axs[1, 1].hist(jisoo_filter.ravel(), 256, [0, 256], color='green', alpha=0.7)
axs[1, 1].set_title('Histogram Citra Setelah Filtering')
axs[1, 1].set_xlabel('Intensitas Piksel')
axs[1, 1].set_ylabel('Frekuensi')
# Menyusun tata letak subplot secara otomatis agar lebih rapi.
plt.tight_layout()
plt.show()

#=========================================================================================================================
#2. HIGH-PASS FILTERING
#=========================================================================================================================
# Baca citra awal
img = cv2.imread('gun.jpg', 0)

# Menerapkan algoritma high-pass filtering:
# Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Sobel dengan ukuran kernel 5
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# Perbesar ukuran hasil plotting
plt.rcParams["figure.figsize"] = (10, 5)

# Menampilkan hasil filter
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 0].axis('off')

axs[0, 1].imshow(laplacian, cmap='gray')
axs[0, 1].set_title('Laplacian')
axs[0, 1].axis('off')

axs[0, 2].imshow(sobelx, cmap='gray')
axs[0, 2].set_title('Sobel X')
axs[0, 2].axis('off')

axs[1, 0].imshow(sobely, cmap='gray')
axs[1, 0].set_title('Sobel Y')
axs[1, 0].axis('off')

axs[1, 1].hist(img.ravel(), 256, [0, 256])
axs[1, 1].set_title('Histogram - Original')

axs[1, 2].hist(laplacian.ravel(), 256, [0, 256])
axs[1, 2].set_title('Histogram - Laplacian')

plt.tight_layout()
plt.show()

# Memanggil citra sebagai grayscale (argument 0)
img = cv2.imread('kratos.jpg', 0)

# Memanggil fungsi Canny Edges dengan argument (citra, nilai_min, nilai_max)
edges = cv2.Canny(img, 100, 200)

# Perbesar ukuran hasil plotting
plt.rcParams["figure.figsize"] = (13, 6)

# Menampilkan citra asli dan citra tepi dalam satu plot
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# Menampilkan histogram citra asli
plt.subplot(2, 2, 3)
plt.hist(img.ravel(), bins=256, color='red')
plt.title('Histogram - Original Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Menampilkan histogram citra sesudah filtering
plt.subplot(2, 2, 4)
plt.hist(edges.ravel(), bins=256, color='red')
plt.title('Histogram - Edge Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

#=========================================================================================================================
#3. TRESHOLDING
#=========================================================================================================================
# membaca gambar
img = cv2.imread('gun.jpg', 0)


# Hitungan threshold.
# Perhatikan nilai ambang batas bawah dan atas dari tiap fungsi yang diberikan
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)


# menampilkan hasil
titles = ['Gambar asli', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# menampilkan beberapa gambar sekaligus
for i in range(6):
    # 3 baris, 2 kolom
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
# Mengatur tata letak subplot agar lebih rapi dan menampilkan plot.
plt.tight_layout()
plt.show()

# Membuat histogram dari gambar asli
plt.subplot(2, 3, 1)
plt.hist(img.ravel(), 256, [0, 256])
plt.title('Histogram Gambar asli')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')


# Membuat histogram dari gambar setelah di filter
for i in range(1, 6):
    plt.subplot(2, 3, i + 1)
    plt.hist(images[i].ravel(), 256, [0, 256])
    plt.title('Histogram ' + titles[i])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()