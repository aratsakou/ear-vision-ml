FROM europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-17.py310:latest

WORKDIR /root

# Copy project files
COPY . /root/ear-vision-ml

# Install dependencies
WORKDIR /root/ear-vision-ml
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install .

# Set python path
ENV PYTHONPATH="${PYTHONPATH}:/root/ear-vision-ml/src"

# Default entrypoint (can be overridden)
ENTRYPOINT ["/bin/bash"]
