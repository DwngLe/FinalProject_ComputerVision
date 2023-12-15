import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh từ thư mục
image = cv2.imread('./data/images/edgeflower.jpg')

# Chuyển hình ảnh sang ảnh xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bộ lọc Prewitt ngang
prewittx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
# Bộ lọc Prewitt dọc
prewitty = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Áp dụng bộ lọc Prewitt ngang
filteredx = cv2.filter2D(gray, -1, prewittx)

# Áp dụng bộ lọc Prewitt dọc
filteredy = cv2.filter2D(gray, -1, prewitty)

# Tính toán độ lớn của gradient
magnitude = np.sqrt(filteredx**2.0 + filteredy**2.0)

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
plt.imshow(filteredx, cmap='gray')
plt.title('Prewitt X')

plt.subplot(1, 5, 4)
plt.imshow(filteredy, cmap='gray')
plt.title('Prewitt Y')

plt.subplot(1, 5, 5)
plt.imshow(normalized, cmap='gray')
plt.title('Prewitt Edge Detection')

plt.show()
