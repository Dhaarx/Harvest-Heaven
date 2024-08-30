import io
import os
from flask import Flask,request,jsonify
from flask_cors import CORS
import numpy as np
from io import BytesIO
import base64
from PIL import Image
# for model
from torchvision import transforms # type: ignore
import torchvision.transforms # type: ignore
import timm # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from typing import List, Tuple
#mongodb
from pymongo import MongoClient # type: ignore
#rag
from PyPDF2 import PdfReader #type: ignore
#rag image
import fitz  # type: ignore # PyMuPDF
from transformers import T5ForConditionalGeneration, T5Tokenizer # type: ignore


app=Flask(__name__)
CORS(app,origins=['*'],methods=['POST','GET'],headers=['Content-Type'])

#device(GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#cropdiseases
class_names =['Grape__Esca(Black_Measles)', 'Tomato__Tomato_mosaic_virus', 'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry_healthy', 'Squash_Powdery_mildew', 'Raspberry_healthy', 'Apple_Apple_scab', 'Pepper,_bell_healthy', 'Apple_Cedar_apple_rust', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Corn(maize)Northern_Leaf_Blight', 'Potato_Early_blight', 'Tomato_healthy', 'Tomato_Septoria_leaf_spot', 'Strawberry_Leaf_scorch', 'Peach_Bacterial_spot', 'Corn(maize)Common_rust', 'Tomato__Leaf_Mold', 'Cherry(including_sour)Powdery_mildew', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 'Grape_Black_rot', 'Pepper,_bell_Bacterial_spot', 'Tomato_Early_blight', 'Peach_healthy', 'Tomato_Late_blight', 'Apple_healthy', 'Apple_Black_rot', 'Cherry(including_sour)healthy', 'Soybean_healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Corn_(maize)healthy', 'Tomato_Target_Spot', 'Potato_Late_blight', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Potato_healthy', 'Tomato_Bacterial_spot', 'Blueberry__healthy']

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

class ResNet9(nn.Module):
    def __init__(self, num_classes):
        super(ResNet9, self).__init__()

        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        # First residual block
        self.res1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        # Second residual block
        self.res2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        )

        # Classifier block
        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Forward pass through initial convolution layers
        x = self.conv1(x)
        x = self.conv2(x)

        # First residual block
        x = x + self.res1(x)

        # Further convolutional layers
        x = self.conv3(x)
        x = self.conv4(x)

        # Second residual block
        x = x + self.res2(x)

        # Final classifier
        x = self.classifier(x)
        return x

# Example usage
model = ResNet9(num_classes=len(class_names))

state_dict = torch.load(r'plant-disease-model.pth', map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict, strict=False)


transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
 ])

#prediction
def pred(model: torch.nn.Module,image_path: bytes,class_names: List[str],image_size: Tuple[int, int] = (299, 299),transform: torchvision.transforms = None,device: torch.device=device):
    
    img = Image.open(BytesIO(image_path))

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    model.to(device)

    model.eval()
    with torch.inference_mode():
      transformed_image = image_transform(img).unsqueeze(dim=0)

      target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    output =class_names[target_image_pred_label]
    return output

#mongodb
mongourl = "mongodb+srv://nmdharineesh2004:mongodb123@cluster0.1ebq3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
try:
    client = MongoClient(mongourl)
    db = client.get_database('herbal_app')
    herb_collection = db['herbs']
    # Test the connection by attempting to retrieve one document
    test_doc = herb_collection.find_one()
    if test_doc:
        print("Successfully connected to MongoDB.")
    else:
        print("Connected to MongoDB, but the collection is empty.")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    
# Function to extract text from PDF
def extract_pdf_content(pdf_path):
    reader = PdfReader(pdf_path)
    extracted_text = []
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text.append(text)
    
    return extracted_text

# Provide the path to the PDF file here
pdf_path = "datasfinal1.pdf"
extracted_text = extract_pdf_content(pdf_path)
pdf1_path = r"datasfinal1.pdf"

#function to fecth image from pdf
def extract_images_from_pdf_based_on_query(pdf1_path, query=None):
    pdf_document = fitz.open(pdf1_path)
    matching_images = []

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        text = page.get_text("text")

        if query is None or query.lower() in text.lower():
            image_list = page.get_images(full=True)
            print(f"Images found on page {page_number + 1}: {len(image_list)}")  # Log the number of images found
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                matching_images.append(img_base64)
                   
    pdf_document.close()
    return matching_images

# Load models and tokenizers
crop_model = T5ForConditionalGeneration.from_pretrained("cropbot")
crop_tokenizer = T5Tokenizer.from_pretrained("cropbot")

livestock_model = T5ForConditionalGeneration.from_pretrained("livestockbot")
livestock_tokenizer = T5Tokenizer.from_pretrained("livestockbot")

#routing,api

@app.route('/')
def home():
    return 'Welcome to the backend server of herbal identifier'

@app.route('/predict',methods=['POST'])
def predict():
    data =request.get_json()
    base64img =data.get('image')

    if not base64img:
        return jsonify({'error':'No image data found'}),400
        
    try:
        image_data = base64.b64decode(base64img)
        op =pred(model=model,image_path=image_data,class_names=class_names,transform=transform,image_size=((256,256)))
        
        matches = [page for page in extracted_text if op.lower() in page.lower()]
        images = extract_images_from_pdf_based_on_query(pdf1_path,op)

        if matches:
            return jsonify({"status":"ok","predicted_class":op,"result": matches[0],"images": images})
        else:
            return jsonify({"status": "error", "message": "Herb not found"}), 404
        
    except Exception as e:
        print(f"Error processing image :{e}")
        return jsonify({'error':'Failed to process image'}),500    

@app.route('/crops',methods=['POST'])
def crops():
    data = request.get_json()
    print("Received data:", data)  

    if not data or 'message' not in data:
        return jsonify({"error": "Invalid input"}), 400

    query = data.get('message', '')
    print("Received query:", query)  

    if not query:
        return jsonify({"error": "Empty query"}), 400

    inputs = crop_tokenizer(query, return_tensors="pt", max_length=128, truncation=True)
    print("Tokenized inputs:", inputs) 
    with torch.no_grad():
        outputs = crop_model.generate(inputs["input_ids"], max_new_tokens=50)
        print("Raw outputs:", outputs)  
    
    answer = crop_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Decoded answer:", answer)  
    
    return jsonify({"response": answer}), 200


@app.route('/livestock', methods=['POST'])
def livestock():
    try:
        data = request.get_json()
        print("Received data:", data)

        if not data or 'message' not in data:
            return jsonify({"error": "Invalid input"}), 400

        query = data.get('message', '')
        print("Received query:", query)  

        if not query:
            return jsonify({"error": "Empty query"}), 400

        inputs = livestock_tokenizer(query, return_tensors="pt", max_length=128, truncation=True)
        print("Tokenized inputs:", inputs)  

        with torch.no_grad():
            outputs = livestock_model.generate(inputs["input_ids"], max_new_tokens=50)
            print("Raw outputs:", outputs)  
        answer = livestock_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Decoded answer:", answer)  

        return jsonify({"response": answer}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Something went wrong on the server."}), 500


if __name__ == '__main__':
    app.run(debug=True ,host='0.0.0.0')