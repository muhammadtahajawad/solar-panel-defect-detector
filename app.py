import gradio as gr
from ultralytics import YOLO

model = YOLO("best.pt")

def predict(image, confidence):
    results = model.predict(image, conf=confidence)[0]
    annotated = results.plot()
    annotated_rgb = annotated[:, :, ::-1]
    num_detections = len(results.boxes)
    return annotated_rgb, f"Detections: {num_detections}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload a solar panel image"),
        gr.Slider(0.05, 0.9, value=0.15, step=0.05, label="Confidence threshold")
    ],
    outputs=[
        gr.Image(type="numpy", label="Detected defects"),
        gr.Textbox(label="Summary")
    ],
    title="Solar Panel Defect Detector (YOLOv8)",
    description=(
        "Upload a solar panel image to detect Bird-Drop, Dusty, Defective, or Clean regions. "
        "Best results on aerial/thermal/electroluminescence-style imagery (the training domain) "
        "— performance on standard ground-level color photos is limited. Lower the confidence "
        "slider if no detections appear."
    )
)

if __name__ == "__main__":
    demo.launch()
