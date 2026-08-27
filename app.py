import gradio as gr
from fastai.vision.all import *
import os

learn = load_learner('model.pkl')

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="My Image Classifier"
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))