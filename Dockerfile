# Dockerfile for modernpmc
##########################
# The Docker image should be built via docker compose

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    curl \
    git \
    python3 \
    python3-dev \
    python3-venv \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-xetex \
    latexmk \
    libsdl2-dev \
    libxml2-dev \
    libxslt-dev \
    graphviz
# texlive, texlive-xetex and latexmk are required for the LaTeX export
# libsdl2 is needed by stormvogel dependency pygame
# libxml2-dev and lixstl-dev are needed by stormvogel dependency lxml

# Install Node.js >=20 required by Myst (Ubuntu 24.04 has older version)
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set-up virtual environment
WORKDIR /app
ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
