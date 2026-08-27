import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import gradio as gr
from fastai.vision.all import *
import os
def is_cat(x): return x[0].isupper() 
learn = load_learner('model.pkl')

def predict(img):
    img = PILImage.create(img)
    img = img.resize((192,192))
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="My Image Classifier"
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))