import os
import sys
import torch
import shutil
import time
import numpy as np
import cv2

# Add SadTalker to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SADTALKER_PATH = os.path.join(BASE_DIR, "SadTalker")
sys.path.append(SADTALKER_PATH)

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

class SproutSadTalkerEngine:
    def __init__(self, checkpoint_dir, avatars_dir, device='cuda', size=256):
        print(f"🚀 INITIALIZING SADTALKER ENGINE ON {device.upper()}...")
        self.device = device
        self.avatars_dir = avatars_dir
        self.checkpoint_dir = checkpoint_dir
        self.size = size
        
        # Initialize paths
        config_dir = os.path.join(SADTALKER_PATH, 'src', 'config')
        self.sadtalker_paths = init_path(checkpoint_dir, config_dir, size=size, old_version=False, preprocess='crop')
        
        # Initialize models
        print("📦 Loading SadTalker Models (Preprocess, Audio2Coeff, Animate)...")
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)
        
        # Cache for preprocessed avatar data (3DMM coeffs, cropped face)
        self.avatar_cache = {}
        self._cache_all_avatars()

    def _cache_all_avatars(self):
        print(f"📂 Pre-processing avatars for SadTalker: {self.avatars_dir}")
        valid_exts = ['.jpg', '.jpeg', '.png']
        if not os.path.exists(self.avatars_dir):
            os.makedirs(self.avatars_dir)
            
        files = [f for f in os.listdir(self.avatars_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        
        # Create temp dir for preprocessing
        temp_preproc_dir = os.path.join(BASE_DIR, "temp_sadtalker_preproc")
        os.makedirs(temp_preproc_dir, exist_ok=True)

        for filename in files:
            path = os.path.join(self.avatars_dir, filename)
            print(f"   👤 Extracting 3DMM for: {filename}...")
            
            try:
                # SadTalker's preprocess extracts 3DMM coefficients from a single image
                # We save this so we don't have to re-extract for every chat response
                save_dir = os.path.join(temp_preproc_dir, filename.replace(".", "_"))
                os.makedirs(save_dir, exist_ok=True)
                
                # generate(pic_path, save_dir, preprocess_type, source_image_flag, pic_size)
                first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
                    path, save_dir, 'crop', source_image_flag=True, pic_size=self.size
                )
                
                if first_coeff_path:
                    self.avatar_cache[filename] = {
                        "first_coeff_path": first_coeff_path,
                        "crop_pic_path": crop_pic_path,
                        "crop_info": crop_info,
                        "source_image": path
                    }
                    print("     ✅ Success!")
                else:
                    print("     ❌ Failed to extract face coefficients.")
            except Exception as e:
                print(f"     ❌ Error: {e}")

    def infer(self, audio_path, output_path, avatar_filename, enhancer='gfpgan', still=False):
        """
        Generates a lively talking head video from audio and a source image.
        Default: still=False (enables natural head movement).
        """
        if avatar_filename not in self.avatar_cache:
            if not self.avatar_cache:
                raise RuntimeError("No avatars cached in SadTalker engine.")
            avatar_filename = list(self.avatar_cache.keys())[0]

        print(f"🎬 Generating Lively Video with SadTalker for {avatar_filename}...")
        data = self.avatar_cache[avatar_filename]
        
        # Use a temporary directory for this generation session
        session_id = str(int(time.time()))
        save_dir = os.path.join(BASE_DIR, f"temp_gen_{session_id}")
        os.makedirs(save_dir, exist_ok=True)

        try:
            # 1. Audio to Coefficients (pose and expression)
            # still=False means head will move naturally
            batch = get_data(data["first_coeff_path"], audio_path, self.device, ref_eyeblink_coeff_path=None, still=still)
            coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style=0, ref_pose_coeff_path=None)
            
            # 2. Coefficients to Video batches
            # size is the output resolution (usually 256 or 512)
            facerender_data = get_facerender_data(
                coeff_path, data["crop_pic_path"], data["first_coeff_path"], audio_path, 
                batch_size=2, input_yaw_list=None, input_pitch_list=None, input_roll_list=None,
                expression_scale=1.0, still_mode=still, preprocess='crop', size=self.size
            )
            
            # 3. Final Face Rendering
            # Resulting video path
            result_video_path = self.animate_from_coeff.generate(
                facerender_data, save_dir, data["source_image"], data["crop_info"], 
                enhancer=enhancer, background_enhancer=None, preprocess='crop', img_size=self.size
            )
            
            # 4. Move to final output path
            if os.path.exists(result_video_path):
                shutil.move(result_video_path, output_path)
                print(f"✅ Video Generation Complete: {output_path}")
            else:
                raise FileNotFoundError("SadTalker failed to produce output video.")
                
        finally:
            # Cleanup session directory
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)

if __name__ == "__main__":
    # Test block
    chkpt = os.path.join(BASE_DIR, "SadTalker", "checkpoints")
    avs = os.path.join(BASE_DIR, "avatars")
    engine = SproutSadTalkerEngine(chkpt, avs)
    # engine.infer("response.mp3", "test_lively.mp4", "womantutor.jpg")
