# rhlf-book

Following: [Deep Reinforcement Learning with Python](https://www.amazon.com/Deep-Reinforcement-Learning-Python-Chatbots/dp/B0CVDQ1HVP#customerReviews)
Book Code at: <https://github.com/nsanghi/drl-2ed>

## Description
This repository contains the code for a book on deep reinforcement learning with Python.

## Installation Instructions

### Installing Homebrew (if not already installed)

Run the following command in your terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Installing required packages

Run the following commands to install the necessary packages using Homebrew:

```sh
brew install swig cmake ffmpeg freeglut git-lfs

git lfs install
```

### Managing Python versions with `asdf`

You can use `asdf` to manage multiple Python versions on your system. First, add it to your shell configuration file:

```bash
echo -e "\n. $(brew --prefix asdf)/asdf.sh" >> ~/.zshrc
source ~/.zshrc

# Install Python plugin for asdf
asdf plugin-add python

# Install Python 3.10
asdf install python 3.10.0

# Set Python 3.10 as the global version
asdf global python 3.10.0
```

### Installing Python dependencies with `uv`

You can use `uv` to manage Python packages. First, add it to your shell configuration file:

```bash
asdf plugin add uv https://github.com/ddzero2c/asdf-uv.git

# Install latest uv version
asdf install uv latest

# Set it as global
asdf global uv latest

# Install Python dependencies
uv sync
```
