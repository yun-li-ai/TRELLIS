This documents Applied’s customization to the TRELLIS repo.

### Which Dockerfile should I use?
- **`Dockerfile`**: Builds the full image from scratch (slow). Avoid unless you need to rebuild dependencies/base.
- **`Dockerfile.update_app`**: Fast “app-only” rebuild. Use when you only changed `headless_app.py` and/or `trellis/utils/postprocessing_utils.py`.

### Build, run, test, and push (Quay)
#### 1) Log in to Quay
Follow the internal instructions: [Quay.io Docker login](https://appliedintuition.atlassian.net/wiki/spaces/eng/pages/110592062/Set+up+Docker#Quay.io-Docker-login).

#### 2) Build the image
`Dockerfile.update_app` builds a new image on top of an existing base image (configured in `Dockerfile.update_app` via `FROM ...`).

Choose a version/tag (example: `1.0.20`), then build:

```bash
docker build -f Dockerfile.update_app -t quay.io/applied_dev/applied2:trellis-1.0.20 .
```

#### 3) Run locally (GPU)
This image starts the FastAPI server from `headless_app.py` on container port `8000`.

```bash
docker run --rm -it --gpus all -p 8000:8000 quay.io/applied_dev/applied2:trellis-1.0.20
```

#### 4) Health check
```bash
curl http://localhost:8000/health
```

#### 5) Test the service (example request)
See `request_test.py`. By default the service is expected at `http://localhost:8000`.

```bash
python request_test.py
```

#### 6) Push to Quay
```bash
docker push quay.io/applied_dev/applied2:trellis-1.0.20
```