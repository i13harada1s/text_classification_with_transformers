FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04 as base

ENV HOME=/root \
    WORKSPACE=/workspace
ENV PYTHON_VERSION=3.9.12

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        make \
        cmake \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libncursesw5-dev \
        llvm \
        wget \
        curl \
        git \
        tzdata \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

FROM base as builder
WORKDIR ${HOME}
# install pyenv
ENV PYENV_ROOT=${HOME}/.pyenv
ENV PATH=${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}
RUN git clone https://github.com/pyenv/pyenv.git ${PYENV_ROOT}
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}
# install poetry
ENV POETRY_ROOT=${HOME}/.poetry
ENV PATH=${POETRY_ROOT}/bin:${PATH}
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# build clean development environment
FROM builder as development-base
WORKDIR ${WORKSPACE}

# build reproduced development environment
FROM builder as development
WORKDIR ${WORKSPACE}
COPY poetry.lock pyproject.toml ./
RUN poetry install

# TODO:
# # build production environment
# FROM base as production
# WORKDIR ${WORKSPACE}
# COPY --from=development ${WORKSPACE}/.venv ./
