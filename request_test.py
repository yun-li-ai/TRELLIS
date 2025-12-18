import requests
import uuid

HOST_PORT = 8000

def call_create_3d_model_service(body):
    url = f"http://localhost:{HOST_PORT}/create-3d-model-from-paths"

    response = requests.post(url, json=body)

    # Check the response
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print("Failed to call service:", response.status_code, response.text)


def call_extract_glb_service(job_id, mesh_simplify=0.95, texture_size=1024):
    url = f"http://localhost:{HOST_PORT}/extract-glb/{job_id}"  # Adjust the URL if your service is hosted elsewhere

    # Prepare the query parameters
    params = {"mesh_simplify": mesh_simplify, "texture_size": texture_size}

    # Make the request
    response = requests.post(url, params=params)

    # Check the response
    if response.status_code == 200:
        # Save the GLB file
        glb_filename = f"{job_id}.glb"
        with open(glb_filename, "wb") as f:
            f.write(response.content)
        print(f"GLB file saved as {glb_filename}")
    else:
        print("Failed to call service:", response.status_code, response.json())


# Run the function
if __name__ == "__main__":
    body = {
        "image_paths": [
            # Note, those should be the path in the container, you will need to
            # 1. Manually copy some test image into the container
            # 2. Change the path to the path in the container
            # 3. Run   `python request_test.py` from the host machine
            "/home/yun/Downloads/3d/teslacybertrucknypd3dsmodel025.jpg",
            "/home/yun/Downloads/3d/teslacybertrucknypd3dsmodel014.jpg",
            "/home/yun/Downloads/3d/teslacybertrucknypd3dsmodel001.jpg",
        ],
        "job_id": str(uuid.uuid4())[:26],
        "seed": 42,
        "randomize_seed": False,
        "ss_guidance_strength": 7.5,
        "ss_sampling_steps": 12,
        "slat_guidance_strength": 3.0,
        "slat_sampling_steps": 12,
    }

    call_create_3d_model_service(body)
    # call_extract_glb_service("c0413825-5812-47fc-b9e6-8315df982efc")
