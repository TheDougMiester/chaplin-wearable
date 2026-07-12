# Chaplin

The purpose of this fork of Amanvir Parhar's Chaplin code is to create a wearable version for people who are unable to speak, but can move their lips (throat cancer survivors, for example). I am running the code on a Jetson Orin Nano Developer Kit, running Jet Pack Version: 6.2.2+b24 (a Raspberry Pi is too slow and memory limited). I was using an IMX519 as my camera, mounted on CAM0. Because of issues with torch, I had to create a container using https://github.com/dusty-nv/jetson-containers; I run chaplin within that container.

At the moment, this thing isn't even beta - but I'm working on it. If you want something more stable, start with Parhar's code. His code, while excellent, wasn't intended for this type of use. If you have any ideas on how to improve what I'm up to, please drop me a line.

At present, I'm running Jetson as a headless setup. You'll need to SSH in from a Linux or Windows machine to run it (I've run it on both). You'll need TightVnC or something running to see what's going on. 

-Doug

Parhar's comments on the original are below (with minor modifications):

![Chaplin Thumbnail](./thumbnail.png)

A visual speech recognition (VSR) tool that reads your lips in real-time and types whatever you silently mouth. Runs fully locally.

Relies on a [model](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages?tab=readme-ov-file#autoavsr-models) trained on the [Lip Reading Sentences 3](https://mmai.io/datasets/lip_reading/) dataset as part of the [Auto-AVSR](https://github.com/mpc001/auto_avsr) project.

Watch a demo of Chaplin [here](https://youtu.be/qlHi0As2alQ).

## Setup

1. Clone the repository, and `cd` into it:
   ```sh
   git clone https://github.com/TheDougMiester/chaplin
   copy the files build_chaplin_env.sh and start_chaplin_container.sh into your $HOME directory 
   cd $HOME
   ```
2. Build the Chaplin environment
   ./build_chaplin_env.sh
   ```
   ...which should download the required model files from Hugging Face Hub and place them in the appropriate directories:
   ```
   chaplin/
   ├── benchmarks/
       ├── LRS3/
           ├── language_models/
               ├── lm_en_subword/
           ├── models/
               ├── LRS3_V_WER19.1/
   ├── ...
   ```
3. Install and run `ollama`, and pull the [`qwen3:4b`](https://ollama.com/library/qwen3:4b) model.
4. Install [`uv`](https://github.com/astral-sh/uv).

## Usage

1. Run the following command: . $HOME/start_chaplin_container.sh - this will start the Jetson container.
2. Once in the container, run root@Jetson:/workspace/chaplin# CHAPLIN_VNC=1 MEDIAPIPE_DETECT_EVERY=8 source /workspace/chaplin/run_chaplin_workspace.sh
3. Once the messages show the camera is running (and the voice says Chaplin is ready), start your VNC viewer on your local machine (TightVNC or whatever)
4. Once the camera feed is displayed, you can start "recording" by pressing the `option` key (Mac) or the `alt` key (Windows/Linux), and start mouthing words.
5. To stop recording, press the `option` key (Mac) or the `alt` key (Windows/Linux) again. The raw VSR output will get logged in your terminal, and the LLM-corrected version will be typed at your cursor.
6. To exit gracefully, focus on the window displaying the camera feed and press `q`.
