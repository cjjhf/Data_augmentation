import random
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np

number = 5

def sharpen_image(image,top_left, bottom_right):

    roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # 创建一个锐化核
    kernel = np.array([[0, -1, 0],
                       [-1, 2, -1],
                       [0, -1, 0]])

    # 将锐化核应用于图像
    sharpened_image = cv2.filter2D(roi, -1, kernel)

    image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = sharpened_image

    # 显示结果
    return image

def move_bounding_boxes(xml_file, image_folder, output_file):
    tree = ET.parse(xml_file)
    img_name = (os.path.splitext(os.path.basename(xml_file))[0])
    image_name = os.path.join(image_folder, '%s.jpg' % img_name)
    output_file = os.path.join(output_file, '%s.jpg' % img_name)
    img = cv2.imread(image_name)
    w, h, _ = img.shape
    b = np.zeros((w, h, 8), dtype=np.uint8)
    for i in range(8):
        b[:, :, i] = 2 ** i
    bit_img = np.zeros((w, h, 3), dtype=np.uint8)
    temp = np.zeros((w, h, 3), 'uint8')

    root = tree.getroot()
    for i in range(1, number):
        bit_img[:, :, 0] = cv2.bitwise_and(img[:, :, 0], b[:, :, i])
        bit_img[:, :, 1] = cv2.bitwise_and(img[:, :, 1], b[:, :, i])
        bit_img[:, :, 2] = cv2.bitwise_and(img[:, :, 2], b[:, :, i])

        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            top_left = (xmin, ymin)
            bottom_right = (xmax, ymax)

            bit_img[:, :, 0] = sharpen_image(bit_img[:, :, 0], top_left, bottom_right)
            bit_img[:, :, 1] = sharpen_image(bit_img[:, :, 1], top_left, bottom_right)
            bit_img[:, :, 2] = sharpen_image(bit_img[:, :, 2], top_left, bottom_right)

        temp[:, :, 0] = cv2.bitwise_or(temp[:, :, 0], bit_img[:, :, 0])
        temp[:, :, 1] = cv2.bitwise_or(temp[:, :, 1], bit_img[:, :, 1])
        temp[:, :, 2] = cv2.bitwise_or(temp[:, :, 2], bit_img[:, :, 2])

        m = bit_img[:, :] > 0
        bit_img[m] = 255

    for i in range(number, 8):
        bit_img[:, :, 0] = cv2.bitwise_and(img[:, :, 0], b[:, :, i])
        bit_img[:, :, 1] = cv2.bitwise_and(img[:, :, 1], b[:, :, i])
        bit_img[:, :, 2] = cv2.bitwise_and(img[:, :, 2], b[:, :, i])

        temp[:, :, 0] = cv2.bitwise_or(temp[:, :, 0], bit_img[:, :, 0])
        temp[:, :, 1] = cv2.bitwise_or(temp[:, :, 1], bit_img[:, :, 1])
        temp[:, :, 2] = cv2.bitwise_or(temp[:, :, 2], bit_img[:, :, 2])

        m = bit_img[:, :] > 0
        bit_img[m] = 255

    cv2.imwrite(output_file,temp)
    print(output_file)

def batch_move_bounding_boxes(xml_folder,image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹

    for filename in os.listdir(xml_folder):
        if filename.lower().endswith(".xml"):

            xml_file = os.path.join(xml_folder, filename)
            move_bounding_boxes(xml_file, image_folder, output_folder)

# 示例用法
xml_folder = r'D:\paper\kitti-voc\1'
image_folder = r'D:\paper\kitti-voc\2'
output_folder = r'E:\process_dataset\kitti\kitti_sharpen\JPEGImages'  # 输出文件夹路径
batch_move_bounding_boxes(xml_folder, image_folder, output_folder)