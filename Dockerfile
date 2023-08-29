FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Update system
RUN apt update && apt install -y python3-pip python3-venv

#Install graphical tool to avoid interaction
RUN apt update && apt install -y libwebkit2gtk-4.0-dev

# Install requirements.txt
COPY ./requirements.txt /api/requirements.txt
WORKDIR /api
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Copy FastAPI app
COPY ./fastapi /api/fastapi
WORKDIR /api/fastapi

# Expose port for FastAPI app to run on
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--env-file", ".env"]
