import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh từ thư mục
image = cv2.imread('./data/images/edgeflower.jpg')

# Chuyển hình ảnh sang ảnh xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Áp dụng bộ lọc Canny
edges = cv2.Canny(gray, 50, 150)

# Áp dụng bộ lọc Laplace
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Hiển thị tất cả các hình ảnh
plt.figure(figsize=(15, 15))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 4, 3)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')

plt.subplot(1, 4, 4)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian Edge Detection')

plt.show()
