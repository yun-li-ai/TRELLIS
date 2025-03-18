import requests

def call_create_3d_model_service(image_paths):
    url = "http://localhost:8000/create-3d-model"  

    # A list of images to create the 3D model.
    image_paths = ["/home/yun/Downloads/yun_li.jpg"]

    # Prepare the files for the request
    files = [("files", (image_path, open(image_path, "rb"), "image/jpeg")) for image_path in image_paths]

    # Make the request without additional data
    response = requests.post(url, files=files)

    # Check the response
    if response.status_code == 200:
        print("Response:", response.json())
        response_json = response.json()
        trial_id = response_json["trial_id"]
        preview_video = response_json["preview_video"]
        print(f"Trial ID: {trial_id}")
        print(f"Preview Video: {preview_video}")
    else:
        print("Failed to call service:", response.status_code, response.text)


def call_extract_glb_service(trial_id, mesh_simplify=0.95, texture_size=1024):
    url = f"http://localhost:8000/extract-glb/{trial_id}"  # Adjust the URL if your service is hosted elsewhere

    # Prepare the query parameters
    params = {
        "mesh_simplify": mesh_simplify,
        "texture_size": texture_size
    }

    # Make the request
    response = requests.post(url, params=params)

    # Check the response
    if response.status_code == 200:
        # Save the GLB file
        glb_filename = f"{trial_id}.glb"
        with open(glb_filename, "wb") as f:
            f.write(response.content)
        print(f"GLB file saved as {glb_filename}")
    else:
        print("Failed to call service:", response.status_code, response.json())


# Run the function
if __name__ == "__main__":
    image_paths = ["path/to/your/image1.jpg", "path/to/your/image2.jpg"]  # Replace with your image paths
    #call_create_3d_model_service(image_paths)
    call_extract_glb_service("151fc89e-ca93-4029-b4e9-4b2de278e7a0")