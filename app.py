import time

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

import draggan
import utils

st.set_page_config(
    page_title="DragGAN Demo",
    page_icon="🐉",
    layout="wide",
)

st.header("🐉 DragGAN")

message_container = st.empty()

col1, col2 = st.columns([1, 3])

def reset():
    st.session_state.clear()

def reset_rerun():
    reset()
    st.experimental_rerun()


### Run/Reset buttons in right col ###
with col2:
    but_col1, but_col2 = st.columns([1,7])
    run_button = but_col1.button("▶️ Run")
    reset_button = but_col2.button("🔁 Reset")


### Settings panel in left col ###
with col1:
    # Models from Self-Distilled SG https://github.com/self-distilled-stylegan/self-distilled-internet-photos
    model_options = {
        "Lions": "https://storage.googleapis.com/self-distilled-stylegan/lions_512_pytorch.pkl",
        "Faces (FFHQ)": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
        "Elephants": "https://storage.googleapis.com/self-distilled-stylegan/elephants_512_pytorch.pkl",
        "Parrots": "https://storage.googleapis.com/self-distilled-stylegan/parrots_512_pytorch.pkl",
        "Horses": "https://storage.googleapis.com/self-distilled-stylegan/horses_256_pytorch.pkl",
        "Bicycles": "https://storage.googleapis.com/self-distilled-stylegan/bicycles_256_pytorch.pkl",
        "Giraffes": "https://storage.googleapis.com/self-distilled-stylegan/giraffes_512_pytorch.pkl",
        "Dogs (1)": "https://storage.googleapis.com/self-distilled-stylegan/dogs_1024_pytorch.pkl",
        "Dogs (2)": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl",
        "Cats": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl",
        "Wildlife": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl",
        "MetFaces": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl",
    }
    model_name = str(st.selectbox("Model", list(model_options.keys()), on_change=reset, help="StyleGAN2 model to use, downloaded and cached on first run"))
    model_url = model_options[model_name]

    target_resolution = int(st.selectbox("Resolution", [256, 512, 1024], index=1, on_change=reset, help="Resize generated image to this resolution (may be different than native model resolution)"))
    seed = st.number_input("Seed", value=22, step=1, min_value=0, on_change=reset, help="Random seed for generating W+ latent")
    truncation_psi = st.number_input("Truncation", value=0.8, step=0.1, min_value=0.0, on_change=reset, help="Truncation trick value to control diversity (higher = more diverse)")
    truncation_cutoff = st.number_input(
        "Truncation Cutoff", value=8, step=1, min_value=-1, max_value=18, on_change=reset, help="Number of layers to apply truncation to (-1 = all layers)"
    )
    n_iter = int(st.number_input("Iterations", value=200, step=5, help="Number of iterations to run optimization", on_change=reset))
    step_size = st.number_input("Step Size", value=1e-3, step=1e-4, min_value=1e-4, format="%.4f", help="Step size (Learning Rate) for gradient descent")
    multiplier = st.number_input("Speed", value=1.0, step=0.05, min_value=0.05, help="Multiplier for target patch movement")
    tolerance = st.number_input("Tolerance", value=2, step=1, min_value=1, help="Number of pixels away from target to stop")
    display_every = st.number_input("Display Every", value=1, step=1, min_value=1, help="Display image during optimization every n iterations")

    if reset_button:
        reset_rerun()

if "points" not in st.session_state:
    st.session_state["points"] = []
    st.session_state["points_types"] = []
    # State variable to track whether the next click should be a 'handle' or 'target'
    st.session_state["next_click"] = "handle"


s = time.perf_counter()
G = draggan.load_model(model_url)

if "W" not in st.session_state:
    W = draggan.generate_W(
        G,
        seed=int(seed),
        truncation_psi=truncation_psi,
        truncation_cutoff=int(truncation_cutoff),
        network_pkl=model_url,
    )
else:
    W = st.session_state["W"]

img, F0 = draggan.generate_image(W, G, network_pkl=model_url)
if img.size[0] != target_resolution:
    img = img.resize((target_resolution, target_resolution))
print(f"Generated image in {(time.perf_counter() - s)*1000:.0f}ms")
draw = ImageDraw.Draw(img)

# Draw an ellipse at each coordinate in points
if "points" in st.session_state and "points_types" in st.session_state:
    for point, point_type in zip(
        st.session_state["points"], st.session_state["points_types"]
    ):
        coords = utils.get_ellipse_coords(point)
        if point_type == "handle":
            draw.ellipse(coords, fill="red")
        elif point_type == "target":
            draw.ellipse(coords, fill="blue")


### Right column image container ###
with col2:
    empty = st.empty()
    with empty.container():
        value = streamlit_image_coordinates(img, key="pil")
        # New point is clicked
        if value is not None:
            point = value["x"], value["y"]
            if point not in st.session_state["points"]:
                # st.session_state["points"].append(point)
                st.session_state["points"].append(point)
                st.session_state["points_types"].append(st.session_state["next_click"])
                st.session_state["next_click"] = (
                    "target" if st.session_state["next_click"] == "handle" else "handle"
                )
                
                st.experimental_rerun()

handles = []
targets = []
if "points" in st.session_state and "points_types" in st.session_state:
    for point, point_type in zip(
        st.session_state["points"], st.session_state["points_types"]
    ):
        if point_type == "handle":
            handles.append(point)
        elif point_type == "target":
            targets.append(point)

## Optimization loop
if run_button:
    if len(handles) > 0 and len(targets) > 0 and len(handles) == len(targets):
        W = draggan.optimize(
            W,
            G,
            handle_points=handles,
            target_points=targets,
            r1=3,
            r2=12,
            tolerance=tolerance,
            max_iter=n_iter,
            lr=step_size,
            multiplier=multiplier,
            empty=empty,
            display_every=display_every,
            target_resolution=target_resolution,
        )
        # st.write(handles)
        # st.write(targets)

        st.session_state.clear()
        st.session_state["W"] = W
        st.experimental_rerun()
    else:
        message_container.warning("Please add at least one handle and one target point.")


