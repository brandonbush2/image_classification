import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

#title
st.title('Leaf Classification')

#header
st.header('Please Upload an image of a Leaf')

#upload file
file = st.file_uploader('', type= ['jpeg', 'jpg', 'png'])

#load classifier
from tensorflow.keras.models import load_model
model = load_model('./inception_lazarus')

from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (256,256)
    image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.new_axis,...]
    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text('Please upload an image file')

else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    predictions = import_and_predict(image,model)
    class_names = { 0: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    1: 'Corn_(maize)___Common_rust_',
                    2: 'Corn_(maize)___Northern_Leaf_Blight',
                    3: 'Corn_(maize)___Northern_Leaf_Blight_oversampled',
                    4: 'Corn_(maize)___Northern_Leaf_Blight_undersampled',
                    5: 'Corn_(maize)___healthy',
                    6: 'Potato___Early_blight',
                    7: 'Potato___Late_blight',
                    8: 'Potato___healthy',
                    9: 'Tomato___Bacterial_spot',
                    10: 'Tomato___Early_blight',
                    11: 'Tomato___Late_blight',
                    12: 'Tomato___Leaf_Mold',
                    13: 'Tomato___Septoria_leaf_spot',
                    14: 'Tomato___Spider_mites Two-spotted_spider_mite',
                    15: 'Tomato___Target_Spot',
                    16: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    17: 'Tomato___Tomato_mosaic_virus',
                    18: 'Tomato___healthy'}
    outs = 'This image is: '+class_names[np.argmax(predictions)]
    st.success(outs)
    






