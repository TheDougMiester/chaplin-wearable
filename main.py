import torch
import hydra
from pipelines.pipeline import InferencePipeline
from chaplin import Chaplin
import sys
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    chaplin = Chaplin()

    # OPEN CAMERA FIRST while GPU memory is free - Argus needs contiguous NVMM
    # Previously we loaded VSR (1GB) then tried to open camera and got
    # NvBufSurfaceFromFd Failed / Failed to create CaptureSession on 8GB Orin Nano
    print("[Main] Opening camera BEFORE loading VSR model to reserve Argus buffers...", flush=True)
    cap, frame_width, frame_height = chaplin._open_camera()
    print(f"[Main] Camera opened: {frame_width}x{frame_height}, now loading VSR model...", flush=True)

    # Now load VSR model - uses remaining memory
    chaplin.vsr_model = InferencePipeline(
        cfg.config_filename, device=torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu"), detector=cfg.detector, face_track=True)

    print("\n MODEL LOADED SUCCESSFULLY! \n")

    # Start webcam loop reusing already-opened camera
    chaplin.start_webcam_with_existing_cap(cap, frame_width, frame_height)

if __name__ == '__main__':
    main()
