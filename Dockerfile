# Base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install dependencies
RUN pip install -r requirements.txt

# Copy the application source code
COPY src /app/src
COPY models /app/models

ENV PYTHONPATH "${PYTHONPATH}:./src"
# Expose the port the app runs on
EXPOSE 7878

WORKDIR /app/src

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7878", "--reload"]
