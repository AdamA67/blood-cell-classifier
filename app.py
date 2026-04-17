import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from model import build_model

# Load model
DEVICE = "cpu"
CLASS_NAMES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 
               'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

model = build_model(num_classes=8, device=DEVICE)
model.load_state_dict(torch.load("outputs/model.pth", map_location=DEVICE))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def classify(image):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Blood Cell Classifier",
    description="Upload a microscopy blood cell image to classify it. Model: ResNet18 fine-tuned via transfer learning. Test accuracy: 88.9%",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()