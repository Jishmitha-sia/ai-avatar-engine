import torch
import numpy as np
import cv2, os, sys, subprocess
from models import Wav2Lip
import audio
import face_detection

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SproutEngine:
    def __init__(self, checkpoint_path, avatars_dir, device='cuda'):
        print(f"\n🚀 INITIALIZING MULTI-AVATAR ENGINE ON {device.upper()}...")
        self.device = device
        self.img_size = 96
        self.batch_size = 128
        self.avatars_dir = avatars_dir
        self.avatar_cache = {}  # Stores processed tensors for each file
        
        # 1. Load Model
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model not found at: {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        
        # 2. Batch Process ALL Avatars
        self._cache_all_avatars()

    def _load_model(self, path):
        model = Wav2Lip()
        print(f"📦 Loading Wav2Lip Model...")
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        s = checkpoint["state_dict"]
        new_s = {k.replace('module.', ''): v for k, v in s.items()}
        model.load_state_dict(new_s)
        return model.to(self.device).eval()

    def _cache_all_avatars(self):
        print(f"📂 Scanning for avatars in: {self.avatars_dir}")
        valid_exts = ['.jpg', '.jpeg', '.png']
        files = [f for f in os.listdir(self.avatars_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        
        if not files:
            print("⚠️ WARNING: No avatar images found! Please add images to backend/avatars/")
            return

        # Initialize Face Detector
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                flip_input=False, device=self.device)

        for filename in files:
            path = os.path.join(self.avatars_dir, filename)
            print(f"   👤 Processing: {filename}...", end="")
            
            try:
                # Load Image
                frame = cv2.imread(path)
                if frame is None: continue

                # Detect Face
                batch_det = detector.get_detections_for_batch(np.array([frame]))
                rect = batch_det[0]
                
                if rect is None:
                    print(" ❌ No face detected. Skipping.")
                    continue

                # Crop Face
                y1, y2, x1, x2 = rect[1], rect[3], rect[0], rect[2]
                pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
                y1 = max(0, y1 - pady1)
                y2 = min(frame.shape[0], y2 + pady2)
                x1 = max(0, x1 - padx1)
                x2 = min(frame.shape[1], x2 + padx2)
                
                cropped_face = frame[y1:y2, x1:x2]
                coords = (y1, y2, x1, x2)

                # Prepare GPU Tensor (The Optimization Step)
                processed_tensor = self._prepare_face_tensor(cropped_face)
                
                # Store in Cache
                self.avatar_cache[filename] = {
                    "full_frame": frame,
                    "coords": coords,
                    "tensor": processed_tensor
                }
                print(" ✅ Cached!")
            except Exception as e:
                print(f" ❌ Error: {e}")

    def _prepare_face_tensor(self, face_img):
        # Resize to 96x96 for Wav2Lip
        face_img = cv2.resize(face_img, (self.img_size, self.img_size))
        
        # Mask lower half
        img_masked = face_img.copy()
        img_masked[:, self.img_size//2:] = 0 
        
        # Stack layers
        img_stacked = np.concatenate((img_masked, face_img), axis=2)
        
        # Convert to Tensor (C,H,W)
        img_tensor = torch.FloatTensor(img_stacked).permute(2, 0, 1) / 255.
        img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
        
        return img_tensor.to(self.device)

    def infer(self, audio_path, output_path, avatar_filename):
        if avatar_filename not in self.avatar_cache:
            raise ValueError(f"Avatar '{avatar_filename}' not loaded! Available: {list(self.avatar_cache.keys())}")

        # Retrieve cached data
        data = self.avatar_cache[avatar_filename]
        cached_tensor = data["tensor"]
        full_frame = data["full_frame"]
        y1, y2, x1, x2 = data["coords"]
        
        # 1. Audio Processing
        if not audio_path.endswith('.wav'):
            temp_wav = audio_path.replace('.mp3', '_temp.wav')
            subprocess.call(f'ffmpeg -y -i "{audio_path}" -strict -2 "{temp_wav}" -loglevel error', shell=True)
            audio_path = temp_wav

        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        
        # 2. Batching Mels
        mel_chunks = []
        mel_step_size = 16
        fps = 25.0
        mel_idx_multiplier = 80./fps 
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        mel_batches = [mel_chunks[i:i + self.batch_size] for i in range(0, len(mel_chunks), self.batch_size)]
        
        # 3. Video Writer
        temp_avi = output_path.replace('.mp4', '_temp.avi')
        h, w = full_frame.shape[:2]
        out = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
        
        # 4. Inference Loop
        for batch_mels in mel_batches:
            mels_tensor = torch.FloatTensor(np.array(batch_mels)).unsqueeze(1).to(self.device)
            
            # Expand cached face to match audio batch size
            curr_batch_size = len(batch_mels)
            faces_tensor = cached_tensor.expand(curr_batch_size, -1, -1, -1)
            
            with torch.no_grad():
                pred = self.model(mels_tensor, faces_tensor)
            
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p in pred:
                p_resized = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                frame_copy = full_frame.copy()
                frame_copy[y1:y2, x1:x2] = p_resized
                out.write(frame_copy)
                
        out.release()
        subprocess.call(f'ffmpeg -y -i "{audio_path}" -i "{temp_avi}" -strict -2 -q:v 1 "{output_path}" -loglevel error', shell=True)