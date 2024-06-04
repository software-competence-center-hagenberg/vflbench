FROM secretflow/secretflow-anolis8:latest
LABEL org.opencontainers.image.authors="du.nguyen.duy@scch.at"

# Sets the working directory in the container
WORKDIR /workspace

# Copies requirements.txt to the docker container's workspace
COPY requirements.txt .

# Installs the python dependencies from requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]