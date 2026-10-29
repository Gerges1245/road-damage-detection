import os, glob, time
import gradio as gr
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

CSS = """
.gradio-container {max-width: 1200px !important; margin: auto;}
#title h1 {text-align: center; margin-bottom: 0;}
#title p {text-align: center; opacity: .8;}
"""

def detect(img, conf, iou, agnostic):
    if img is None:
        return None, pd.DataFrame(columns=["Damage type", "Count", "Avg confidence"]), "Upload an image to start."
    t = time.perf_counter()
    r = model.predict(img, conf=conf, iou=iou, agnostic_nms=agnostic, imgsz=640, verbose=False)[0]
    ms = (time.perf_counter() - t) * 1000
    annotated = r.plot(line_width=3)[:, :, ::-1]          # BGR -> RGB

    if len(r.boxes) == 0:
        table = pd.DataFrame({"Damage type": ["None detected"], "Count": [0], "Avg confidence": ["-"]})
        return annotated, table, f"**No damage detected** at confidence ≥ {conf:.2f} · {ms:.0f} ms"

    df = pd.DataFrame({"Damage type": [model.names[int(c)] for c in r.boxes.cls],
                       "conf": r.boxes.conf.cpu().numpy()})
    table = (df.groupby("Damage type")
               .agg(Count=("conf", "size"), **{"Avg confidence": ("conf", "mean")})
               .reset_index().sort_values("Count", ascending=False))
    table["Avg confidence"] = table["Avg confidence"].round(2)
    msg = f"**{len(df)} detections** · **{table.shape[0]} damage types** · inference {ms:.0f} ms (CPU)"
    return annotated, table, msg

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path).round(3)
    return pd.DataFrame({"info": [f"{path} not found next to app.py"]})

EXAMPLES = sorted(glob.glob("samples/*.*"))[:12]

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), css=CSS, title="Road Damage Detection") as demo:
    gr.Markdown("# 🛣️ Road Damage Detection\nYOLOv8n fine-tuned on 7 road damage classes", elem_id="title")

    with gr.Tabs():
        with gr.Tab("🔍 Detect"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(type="pil", label="Road image",
                                   sources=["upload", "webcam", "clipboard"], height=420)
                    conf = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="Confidence threshold",
                                     info="Minimum confidence to show a box. Lower = more detections, more false alarms.")
                    iou = gr.Slider(0.1, 0.9, 0.5, step=0.05, label="Overlap threshold (NMS IoU)",
                                    info="Lower = removes more overlapping duplicate boxes.")
                    agn = gr.Checkbox(False, label="Merge overlapping boxes of different classes")
                    with gr.Row():
                        run = gr.Button("Detect", variant="primary")
                        clear = gr.ClearButton(value="Clear")
                with gr.Column():
                    out = gr.Image(label="Detections", height=420)
                    status = gr.Markdown()
                    table = gr.Dataframe(label="Summary", interactive=False)

            ins, outs = [inp, conf, iou, agn], [out, table, status]
            run.click(detect, ins, outs)
            conf.release(detect, ins, outs)
            iou.release(detect, ins, outs)
            clear.add([inp, out, table, status])
            if EXAMPLES:
                gr.Examples([[p, 0.25, 0.5, False] for p in EXAMPLES], inputs=ins, outputs=outs,
                            fn=detect, cache_examples=False, label="Sample images (from the test set)")

        with gr.Tab("📊 Model comparison"):
            gr.Markdown("All five models evaluated on the same test set (681 images). "
                        "Speed was measured on CPU.")
            gr.Dataframe(load_csv("results/final_comparison.csv"), interactive=False)
            with gr.Row():
                if os.path.exists("results/comparison_bars.png"):
                    gr.Image("results/comparison_bars.png", show_label=False, interactive=False)
                if os.path.exists("results/per_class.png"):
                    gr.Image("results/per_class.png", show_label=False, interactive=False)

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
**Model:** YOLOv8n (default hyperparameters), the best model on the test set (mAP50 0.566, mAP50-95 0.346).

**Classes:** alligator, block, crack, edge, longitudinal, pothole, transverse.

**Known limitations**
- `alligator` and `longitudinal` are the weakest classes in every model we tested.
- `edge` appears only once in the validation set and not at all in the test set, so it is effectively untested.
""")

if __name__ == "__main__":
    demo.launch()