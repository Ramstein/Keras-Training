from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import cv2, numpy as np




#prebuilt model eith prebuilt weights on imagenet
model = VGG16(weights='imagenet', include_top=True)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

#resize into VGG16 trained images
im = cv2.resize(cv2.imread('tabby-cat-close-up-portrait-69932.jpeg'), (224, 224))
im = np.expand_dims(im, axis=0)

#predict
out = model.predict(im)

print(np.argmax(out)) #this will print 820 for steaming train
plt.xlabel('prediction')

plt.plot(out.ravel())
plt.show()



