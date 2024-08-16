from load_model import model
import cv2
import pytesseract
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

Image_path = "Test_Images/img.jpg"

model_preds = model.predict(Image_path, confidence=40, overlap=30).json()

print(model_preds)

# visualize your prediction
model.predict(Image_path, confidence=40, overlap=30).save("predictions.jpg")

def extract_plate(image_path, model_preds):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert bounding box dimensions to coordinates
    x_center = int(model_preds['predictions'][0]['x'])
    y_center = int(model_preds['predictions'][0]['y'])
    width = int(model_preds['predictions'][0]['width'])
    height = int(model_preds['predictions'][0]['height'])
    
    # Calculate the bottom-right corner of the bounding box
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    
    # Crop the number plate region
    plate_image = image[y_min:y_max, x_min:x_max]
    
    # Convert to PIL Image and show
    pil_image = Image.fromarray(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
    pil_image.show()  # This will open the cropped image using the default image viewer
    
    return plate_image

plate_image = extract_plate(Image_path, model_preds)

def read_plate_text(plate_image):
    # Convert the image to a PIL Image object
    pil_image = Image.fromarray(plate_image)
    
    # Use pytesseract to do OCR on the image
    text = pytesseract.image_to_string(pil_image, config='--psm 8')
    
    return text

print('The detected number is ', read_plate_text(plate_image))
