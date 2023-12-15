import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh từ thư mục
image = cv2.imread('./data/images/edgeflower.jpg')

# Chuyển hình ảnh sang ảnh xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Áp dụng bộ lọc Sobel ngang
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)

# Áp dụng bộ lọc Sobel dọc
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

# Tính toán độ lớn của gradient
magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)

# Chuẩn hóa đến khoảng [0, 255] và chuyển đổi sang kiểu uint8
normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Hiển thị tất cả các hình ảnh
plt.figure(figsize=(20, 20))

plt.subplot(1, 5, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 5, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 5, 3)
plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X')

plt.subplot(1, 5, 4)
plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y')

plt.subplot(1, 5, 5)
plt.imshow(normalized, cmap='gray')
plt.title('Sobel Edge Detection')

plt.show()
