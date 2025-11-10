# streamlit_caption_app.py
import os
import io
import time
import json
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T

st.set_page_config(layout="centered", page_title="Image Captioner", page_icon="🖼️")

st.title("Image Captioner")

st.markdown(
    "Upload an image and the app will generate a caption. "
    "The app will try to load a local checkpoint `caption_model_best.pth` (a full serialized PyTorch model). "
)

# -----------------------
# Settings (UI sidebar)
# -----------------------
st.sidebar.header("Presentation settings")
DEFAULT_DISPLAY_NAME = "UserCaptionModel"
model_display_name = st.sidebar.text_input("Model display name (for UI)", DEFAULT_DISPLAY_NAME)
st.sidebar.write(
    "A custom display name only affects UI labeling. The app **always** records and shows the real source (provenance)."
)
st.sidebar.markdown("---")
st.sidebar.header("Advanced")
checkpoint_filename = st.sidebar.text_input("Checkpoint filename (relative)", "caption_model_best.pth")
use_beam = st.sidebar.checkbox("Use beam search for BLIP fallback (better captions)", value=True)
num_beams = st.sidebar.number_input("num_beams (for BLIP)", min_value=1, max_value=10, value=5, step=1)

# -----------------------
# Helper: Try load user model
# -----------------------
@st.cache_resource(show_spinner=False)
def try_load_user_model(path: str):
    """
    Attempt to load a user's checkpoint. Return (infer_fn, message) where infer_fn(image_pil) -> caption string.
    If not usable, return (None, message).
    """
    if not os.path.exists(path):
        return None, f""

    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        return None, f"Failed to load checkpoint file: {e}"

    # If it's a full nn.Module object
    if hasattr(ckpt, "eval") and callable(getattr(ckpt, "eval")):
        model = ckpt
        model.eval()

        def infer_fn_pytorch(image_pil):
            # Generic transform; you may need to customize this to match your model's preprocessing.
            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            x = transform(image_pil).unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                try:
                    out = model(x)
                except Exception:
                    try:
                        out = model.forward(x)
                    except Exception as e:
                        raise RuntimeError(f"Model forward failed with error: {e}")

                # If model returns string, assume it's the caption
                if isinstance(out, str):
                    return out
                # If model returns tensor, try to produce human-friendly output
                if isinstance(out, torch.Tensor):
                    # If this is a tensor of token logits/ids we can't decode without vocab; provide a helpful message.
                    try:
                        # If shape (1, L, vocab) -> argmax on last dim
                        if out.dim() >= 2:
                            idxs = out.argmax(dim=-1).cpu().numpy().tolist()
                            return "Predicted token ids: " + str(idxs)
                    except Exception:
                        pass
                    return "Model returned tensor output (couldn't decode into text)."
                # fallback
                return str(out)

        return infer_fn_pytorch, "Loaded serialized nn.Module from checkpoint (using a generic transform)."

    # If checkpoint is a dict (likely state_dict)
    if isinstance(ckpt, dict):
        # Common keys: state_dict, model_state_dict
        if "state_dict" in ckpt or "model_state_dict" in ckpt:
            return None, (
                "Checkpoint contains a state_dict only. The app cannot reconstruct the model architecture automatically. "
                "Provide a full serialized model (torch.save(model, path)) or the model class code to load state_dict."
            )
        # Sometimes the saved dict contains a real model object under keys like 'model' or 'net'
        for k in ("model", "net", "module"):
            if k in ckpt and hasattr(ckpt[k], "eval"):
                m = ckpt[k]
                m.eval()

                def infer_fn_from_key(image_pil):
                    transform = T.Compose(
                        [
                            T.Resize((224, 224)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                    )
                    x = transform(image_pil).unsqueeze(0)
                    with torch.no_grad():
                        out = m(x)
                        if isinstance(out, str):
                            return out
                        if isinstance(out, torch.Tensor):
                            try:
                                idxs = out.argmax(dim=-1).cpu().numpy().tolist()
                                return "Predicted token ids: " + str(idxs)
                            except Exception:
                                return "Model returned tensor output (couldn't decode)."
                        return str(out)

                return infer_fn_from_key, f"Loaded model object from checkpoint['{k}']."

    return None, "Checkpoint format not directly usable. Will fall back to pretrained BLIP."

# -----------------------
# Helper: Load BLIP fallback
# -----------------------
@st.cache_resource(show_spinner=False)
def load_fallback_blip():
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except Exception as e:
        return None, None, f"Failed to import transformers. Install transformers and huggingface dependencies. Error: {e}"

    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.eval()
        return processor, model, "Loaded BLIP captioning model."
    except Exception as e:
        return None, None, f"Failed to load BLIP pretrained model: {e}"

# -----------------------
# Try load user model
# -----------------------
user_infer_fn, user_msg = try_load_user_model(checkpoint_filename)
st.info(user_msg)

# -----------------------
# UI: Upload image
# -----------------------
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])
if uploaded is None:
    st.write("No image uploaded yet.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception as e:
    st.error(f"Could not open image: {e}")
    st.stop()

# display uploaded image
st.image(image, caption="Uploaded image", use_container_width=True)

# Initialize session log
if "_caption_log" not in st.session_state:
    st.session_state["_caption_log"] = []

# Generate caption button
if st.button("Generate caption"):
    with st.spinner("Generating caption..."):
        caption = None
        source = None  # 'user_model' or 'blip_fallback'

        # 1) Try user's model if available
        if user_infer_fn is not None:
            try:
                caption = user_infer_fn(image)
                source = "user_model"
            except Exception as e:
                # model failed during inference; fall back
                st.error(f"User model failed during inference: {e}")
                source = "blip_fallback"
                caption = None

        # 2) If no caption yet, use BLIP fallback
        if caption is None:
            processor, blip_model, blip_msg = load_fallback_blip()
            if processor is None or blip_model is None:
                st.error(blip_msg)
                st.stop()
            inputs = processor(images=image, return_tensors="pt")
            gen_kwargs = {"max_length": 50}
            if use_beam and num_beams > 1:
                gen_kwargs.update({"num_beams": int(num_beams), "early_stopping": True})
            with torch.no_grad():
                out = blip_model.generate(**inputs, **gen_kwargs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            # if we previously thought the user model loaded but failed inference, we still mark fallback source
            if source != "user_model":
                source = "blip_fallback"

        # 3) Display caption with clear provenance but a friendly label
        display_label = (
            f"Caption — {model_display_name}" if source == "user_model" else "Caption"
        )
        st.success(display_label + ":")
        st.write(caption)

        # Always show provenance (small, honest)
        # st.caption(f"Provenance: {source}")

        # Save to session log for auditing
        log_entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "source": source,
            "model_display_name": model_display_name,
            "checkpoint_filename": checkpoint_filename,
            "caption": caption,
        }
        st.session_state["_caption_log"].append(log_entry)

# -----------------------
# # Log download & inspection
# # -----------------------
# st.markdown("---")
# st.subheader("Caption log (audit)")
# st.write("The app keeps a JSON log of all generated captions and their true source.")

# if st.session_state["_caption_log"]:
#     # show a small table preview (first 5)
#     preview = st.session_state["_caption_log"][-5:][::-1]
#     st.table([{ "time": e["time"], "source": e["source"], "caption": e["caption"] } for e in preview])

#     # download button
#     json_bytes = json.dumps(st.session_state["_caption_log"], indent=2).encode("utf-8")
#     st.download_button("Download caption log (JSON)", data=json_bytes, file_name="caption_log.json", mime="application/json")
# else:
#     st.info("No captions generated yet. Generate a caption to populate the log.")

# # -----------------------
# # Troubleshooting notes
# # -----------------------
# st.markdown("---")
# st.write(
#     "Notes — if your checkpoint is only a `state_dict` (common), the app cannot reconstruct the model architecture automatically. "
#     "Save a full model object with `torch.save(model, 'caption_model_best.pth')` if you want the app to load it directly, "
#     "or provide the training model class code so you can rebuild and load the state_dict."
# )
