# Dockerfile for modernpmc
##########################
# The Docker image can be built by executing:
# docker build -t modernpmc .
# Run with as follows and afterwards open localhost:3000
# docker run -p 3000:3000 modernpmc
FROM ubuntu:24.04

# Install dependencies
RUN apt-get update -qq
RUN apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-dev \
    python3-venv \
    nodejs \
    npm \
    texlive \
    texlive-xetex \
    latexmk \
    libsdl2-dev \
    libxml2-dev \
    libxslt-dev \
    graphviz
# nodejs and npm are required by Jupyter
# texlive, texlive-xetex and latexmk are required for the LaTeX export
# libsdl2 is needed by stormvogel dependency pygame
# libxml2-dev and lixstl-dev are needed by stormvogel dependency lxml


# Set-up virtual environment
WORKDIR /app
ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy
COPY . .

# Build
RUN jupyter book build --execute --html
RUN jupyter book build --execute --pdf


## Serve static HTML
WORKDIR /app/_build/html
EXPOSE 3000
CMD ["python3", "-m", "http.server", "3000"]
