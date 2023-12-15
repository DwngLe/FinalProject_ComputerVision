import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào
img = cv2.imread('./data/images/edgeflower.jpg', 0)
img_original = img.copy()  # Lưu lại ảnh gốc để hiển thị sau này

# Binarize ảnh
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Tạo một hình kernel
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Khởi tạo ảnh skeleton
skeleton = np.zeros(img.shape, np.uint8)

# Lặp cho đến khi ảnh hoàn toàn được xử lý
while True:
    # Sử dụng phép toán morphological open
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Tìm sự khác biệt giữa ảnh gốc và ảnh sau khi mở
    temp = cv2.subtract(img, open)

    # Erode ảnh gốc và cập nhật ảnh skeleton
    eroded = cv2.erode(img, kernel)
    skeleton = cv2.bitwise_or(skeleton, temp)
    img = eroded.copy()

    # Nếu không còn pixel nào trong ảnh, thoát khỏi vòng lặp
    if cv2.countNonZero(img) == 0:
        break

# Hiển thị ảnh gốc và ảnh skeleton trên cùng một biểu đồ
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(img_original, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(skeleton, cmap='gray'), plt.title('Skeleton Image')
plt.show()
