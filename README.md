# rhlf-book

Following: [Deep Reinforcement Learning with Python](https://www.amazon.com/Deep-Reinforcement-Learning-Python-Chatbots/dp/B0CVDQ1HVP#customerReviews)
Book Code at: <https://github.com/nsanghi/drl-2ed>

## Recommended Prerequisites

To install the required packages on a Mac using Homebrew, run the following commands in your terminal:

```sh
# Install Homebrew if you haven't already
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install swig cmake ffmpeg freeglut git-lfs

# Initialize Git LFS
git lfs install
```

To manage Python versions, you can use asdf. First, install asdf and then install Python 3.10:

```bash
# Add asdf to your shell
echo -e "\n. $(brew --prefix asdf)/asdf.sh" >> ~/.zshrc
source ~/.zshrc

# Install Python plugin for asdf
asdf plugin-add python

# Install Python 3.10
asdf install python 3.10.0

# Set Python 3.10 as the global version
asdf global python 3.10.0
```

To manage Python packages, you can use uv. First, install uv and then install the required Python dependencies:

```bash
# Add uv plugin to asdf
asdf plugin add uv https://github.com/ddzero2c/asdf-uv.git

# Install latest uv version
asdf install uv latest

# Set it as global
asdf global uv latest

# Install Python dependencies
uv sync
```