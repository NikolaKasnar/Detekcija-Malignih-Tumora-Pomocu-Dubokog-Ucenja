# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory inside container
WORKDIR /app

# Copy all your code
COPY . .

# Install dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN pip install --upgrade pip
RUN pip install \
    numpy pandas matplotlib scikit-learn tqdm ipywidgets albumentations \
    torchsummary pydicom lmdb timm pillow

# Default command
CMD ["python", "nocno_treniranje.py"]
