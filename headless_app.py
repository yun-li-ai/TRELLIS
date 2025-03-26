import os
from typing import *
import torch
import numpy as np
import imageio
import time
from easydict import EasyDict as edict
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

print(f"Using attention backend: {os.environ.get('ATTN_BACKEND', 'default')}")


class Create3DModelRequestModel(BaseModel):
    image_paths: List[str]
    job_id: str
    seed: int = 0
    randomize_seed: bool = True
    ss_guidance_strength: float = 7.5
    ss_sampling_steps: int = 12
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 12


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = "/tmp/sensor-sim-trellis-shared/output"
os.makedirs(TMP_DIR, exist_ok=True)


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """Clean up files older than max_age_hours"""
    current_time = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if (current_time - os.path.getmtime(filepath)) > (max_age_hours * 3600):
                try:
                    os.remove(filepath)
                except OSError:
                    pass


@app.on_event("startup")
async def startup_event():
    """Run cleanup on startup"""
    cleanup_old_files(TMP_DIR)


# Before pipeline initialization
if os.environ.get("ATTN_BACKEND") != "xformers":
    print("Warning: ATTN_BACKEND not set to xformers")

# Initialize pipeline globally
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()


def preprocess_image(
    trial_id: str, image: Image.Image, image_name: str, output_image_paths: List[str]
) -> Tuple[str, Image.Image]:
    processed_image = pipeline.preprocess_image(image)
    image_path = f"{TMP_DIR}/{trial_id}/{image_name}.png"
    processed_image.save(image_path)
    output_image_paths.append(image_path)
    return processed_image


def pack_state(gs: Gaussian, mesh: MeshExtractResult, trial_id: str) -> dict:
    return {
        "gaussian": {
            **gs.init_params,
            "_xyz": gs._xyz.cpu().numpy().tolist(),
            "_features_dc": gs._features_dc.cpu().numpy().tolist(),
            "_scaling": gs._scaling.cpu().numpy().tolist(),
            "_rotation": gs._rotation.cpu().numpy().tolist(),
            "_opacity": gs._opacity.cpu().numpy().tolist(),
        },
        "mesh": {
            "vertices": mesh.vertices.cpu().numpy().tolist(),
            "faces": mesh.faces.cpu().numpy().tolist(),
        },
        "trial_id": trial_id,
    }


@app.post("/create-3d-model-from-paths")
async def create_3d_model_from_paths(request: Create3DModelRequestModel):
    image_paths = request.image_paths
    job_id = request.job_id
    seed = request.seed
    randomize_seed = request.randomize_seed
    ss_guidance_strength = request.ss_guidance_strength
    ss_sampling_steps = request.ss_sampling_steps
    slat_guidance_strength = request.slat_guidance_strength
    slat_sampling_steps = request.slat_sampling_steps

    print(f"Creating 3D model from {len(image_paths)} images")
    os.makedirs(f"{TMP_DIR}/{job_id}", exist_ok=True)

    errors = []
    images = {}
    output_image_paths = []

    # Check if file paths are valid and read images
    for path in image_paths:
        if not os.path.exists(path):
            errors.append(f"File not found: {path}")
            continue
        try:
            image = Image.open(path)
            base_name = os.path.splitext(os.path.basename(path))[0]
            images[base_name] = image
        except Exception as e:
            errors.append(f"Error opening file {path}: {str(e)}")

    if not images:
        return {"error": "No valid images found", "details": errors}

    processed_images = [
        preprocess_image(job_id, image, image_name, output_image_paths)
        for image_name, image in images.items()
    ]

    # Generate 3D model
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)

    outputs = pipeline.run_multi_image(
        processed_images,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )

    # Generate preview video
    video = render_utils.render_video(outputs["gaussian"][0], num_frames=120)["color"]
    video_geo = render_utils.render_video(outputs["mesh"][0], num_frames=120)["normal"]
    video = [
        np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))
    ]
    video_path = f"{TMP_DIR}/{job_id}/preview.mp4"
    imageio.mimsave(video_path, video, fps=15)

    # Pack state and return results
    state = pack_state(outputs["gaussian"][0], outputs["mesh"][0], job_id)

    # Save state file
    with open(f"{TMP_DIR}/{job_id}/state.json", "w") as f:
        json.dump(state, f)

    # Generate and save GLB file directly from state
    glb_path = f"{TMP_DIR}/{job_id}/model.glb"
    gs = Gaussian(
        aabb=state["gaussian"]["aabb"],
        sh_degree=state["gaussian"]["sh_degree"],
        mininum_kernel_size=state["gaussian"]["mininum_kernel_size"],
        scaling_bias=state["gaussian"]["scaling_bias"],
        opacity_bias=state["gaussian"]["opacity_bias"],
        scaling_activation=state["gaussian"]["scaling_activation"],
    )
    gs._xyz = torch.tensor(state["gaussian"]["_xyz"], device="cuda")
    gs._features_dc = torch.tensor(state["gaussian"]["_features_dc"], device="cuda")
    gs._scaling = torch.tensor(state["gaussian"]["_scaling"], device="cuda")
    gs._rotation = torch.tensor(state["gaussian"]["_rotation"], device="cuda")
    gs._opacity = torch.tensor(state["gaussian"]["_opacity"], device="cuda")

    mesh = edict(
        vertices=torch.tensor(state["mesh"]["vertices"], device="cuda"),
        faces=torch.tensor(state["mesh"]["faces"], device="cuda"),
    )

    glb = postprocessing_utils.to_glb(
        gs, mesh, simplify=0.95, texture_size=1024, verbose=False
    )
    glb.export(glb_path)

    return {
        "job_id": job_id,
        "preview_video": video_path,
        "glb_file": glb_path,
        "output_images": output_image_paths,
        "errors": errors,  # Include any errors encountered
    }


@app.get("/preview/{job_id}")
async def get_preview(job_id: str):
    video_path = f"{TMP_DIR}/{job_id}/preview.mp4"
    return FileResponse(video_path)


@app.post("/extract-glb/{job_id}")
async def extract_glb(
    job_id: str, mesh_simplify: float = 0.95, texture_size: int = 1024
):
    # Load the state file
    state_path = f"{TMP_DIR}/{job_id}/state.json"
    if not os.path.exists(state_path):
        return {"error": "Model UUID not found"}

    # Add this line to load the state
    with open(state_path, "r") as f:
        state = json.load(f)

    # Generate GLB
    glb_path = f"{TMP_DIR}/{job_id}/model.glb"
    gs = Gaussian(
        aabb=state["gaussian"]["aabb"],
        sh_degree=state["gaussian"]["sh_degree"],
        mininum_kernel_size=state["gaussian"]["mininum_kernel_size"],
        scaling_bias=state["gaussian"]["scaling_bias"],
        opacity_bias=state["gaussian"]["opacity_bias"],
        scaling_activation=state["gaussian"]["scaling_activation"],
    )
    gs._xyz = torch.tensor(state["gaussian"]["_xyz"], device="cuda")
    gs._features_dc = torch.tensor(state["gaussian"]["_features_dc"], device="cuda")
    gs._scaling = torch.tensor(state["gaussian"]["_scaling"], device="cuda")
    gs._rotation = torch.tensor(state["gaussian"]["_rotation"], device="cuda")
    gs._opacity = torch.tensor(state["gaussian"]["_opacity"], device="cuda")

    mesh = edict(
        vertices=torch.tensor(state["mesh"]["vertices"], device="cuda"),
        faces=torch.tensor(state["mesh"]["faces"], device="cuda"),
    )

    glb = postprocessing_utils.to_glb(
        gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False
    )
    glb.export(glb_path)

    return FileResponse(glb_path, filename=f"{job_id}.glb")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "attention_backend": os.environ.get("ATTN_BACKEND", "default"),
        "gpu_info": torch.cuda.get_device_properties(0).__str__(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
