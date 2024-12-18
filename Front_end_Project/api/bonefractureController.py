import numpy as np
from keras.models import load_model
from keras.preprocessing import image

saved_model = load_model("model/bone_model.h5")

def bonefracture(path):
    img = image.load_img(path, target_size=(300, 300))
   # plt.imshow(img,cmap='Greys')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
  
    images = np.vstack([x])
    classes = saved_model.predict(images,batch_size=None, verbose=0, steps=None, callbacks=None)
    print(classes[0])
    if classes[0]>0.5:
        result='image is positive'
    else:
        result='image is negative'
 
    return result