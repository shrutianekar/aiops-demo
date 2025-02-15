# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY mlops/requirements.txt ./requirements.txt
COPY mlops/model_train_pipeline.py ./model_train_pipeline.py

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run fine-tuning
CMD ["python", "/app/model_train_pipeline.py"]


# Step 1: Build the image
#docker build -t ai-fine-tune:latest .

# Step 2: Tag the image
#docker tag ai-fine-tune:latest shruti1997/ai-fine-tune:latest

# Step 3: Push the image to Docker Hub
#docker push shruti1997/ai-fine-tune:latest
