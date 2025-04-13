import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import svm

folder_0 = "face_data/0/"
folder_1 = "face_data/1/"

def image_to_vector(path, size = (48, 48)):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Can't read image from:{path}")
    else:
        image = cv2.resize(image, size)
        image = image.astype(np.float32) / 255.0
        vector = image.flatten()
    return vector

def import_data(path, size = (48, 48)):
    vectors = []
    images = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        vector = image_to_vector(file_path, size)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
        vectors.append(vector)
    vectors = np.array(vectors)
    return vectors, images

data_0, img_0 = import_data(folder_0)
data_1, img_1 = import_data(folder_1)
label_0 = np.array([0] * 300)
label_1 = np.array([1] * 300)
data_label_0 = np.column_stack((data_0, label_0))
data_label_1 = np.column_stack((data_1, label_1))

def get_training_testing(data):
    data = np.array(data)
    length = data.shape[0]
    mask = np.zeros(length, dtype = bool)
    index = np.random.choice(length, 50, replace = False)
    mask[index] = True
    return data[~mask], data[mask], mask

data_train_0, data_test_0, mask0 = get_training_testing(data_label_0)
data_train_1, data_test_1, mask1 = get_training_testing(data_label_1)
data_train = np.concatenate((data_train_0, data_train_1), axis = 0)
data_test = np.concatenate((data_test_0, data_test_1), axis = 0)
img_train = list(x for x, m in zip(img_0, mask0) if m == 0) + list(x for x, m in zip(img_1, mask1) if m == 0)
img_test = list(x for x, m in zip(img_0, mask0) if m == 1) + list(x for x, m in zip(img_1, mask1) if m == 1)


penal = 0.1
ker = 'linear'

def evaluate_classifier(X, y, penalty, ker):
    model = svm.SVC(C = penalty, kernel = ker)
    model.fit(data_train[:, :-1], data_train[:, -1])
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    return accuracy, model.support_vectors_, model.support_

train_accuracy, support_vectors, support_vector_indices = evaluate_classifier(data_train[:, :-1], data_train[:, -1], penal, ker)
test_accuracy, spvtors, sprtidx = evaluate_classifier(data_test[:, :-1], data_test[:, -1], penal, ker)


print("test accuracy:", test_accuracy)

os.makedirs("Support Vectors", exist_ok=True)

fig, axes = plt.subplots(nrows = 1, ncols = len(support_vector_indices), figsize = (20, 5))
for ax, index in zip(axes, support_vector_indices):
    image = img_train[index]
    ax.imshow(image)
    ax.axis('off')
    #save_path = os.path.join("Support Vectors", f"image_{index}.png")
    #plt.imsave(save_path, image)
plt.show()
