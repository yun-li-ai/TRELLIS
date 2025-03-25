import requests


def call_create_3d_model_service(image_paths):
    url = "http://localhost:8000/create-3d-model-from-paths"

    # Prepare the files for the request
    # files = [("files", (image_path, open(image_path, "rb"), "image/jpeg")) for image_path in image_paths]

    # Make the request without additional data
    # response = requests.post(url, files=files)

    response = requests.post(url, json=image_paths)

    # Check the response
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print("Failed to call service:", response.status_code, response.text)


def call_extract_glb_service(trial_id, mesh_simplify=0.95, texture_size=1024):
    url = f"http://localhost:8000/extract-glb/{trial_id}"  # Adjust the URL if your service is hosted elsewhere

    # Prepare the query parameters
    params = {"mesh_simplify": mesh_simplify, "texture_size": texture_size}

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
    image_paths = [
        "/home/yun/Documents/images/cybertruck/1.png",
        "/home/yun/Documents/images/cybertruck/2.png",
        "/home/yun/Documents/images/cybertruck/3.png",
    ]

    # image_paths = ["/home/yun/Downloads/yun_li.jpg"]
    call_create_3d_model_service(image_paths)
    # call_extract_glb_service("c0413825-5812-47fc-b9e6-8315df982efc")
