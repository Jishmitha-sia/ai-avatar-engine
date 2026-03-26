import torch
import numpy as np
import cv2, os, sys, subprocess
from models import Wav2Lip
import audio
import face_detection

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SproutEngine:
    def __init__(self, checkpoint_path, avatars_dir, device='cuda'):
        print(f"\n🚀 INITIALIZING CORRECTED ENGINE ON {device.upper()}...")
        self.device = device
        # [NEW] Handle Blackwell (RTX 50-series) warning
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
             print("💡 NOTE: RTX 50-series (Blackwell) detected. Using optimized CUDA paths if available.")
        self.img_size = 96
        self.batch_size = 128
        self.avatars_dir = avatars_dir
        self.avatar_cache = {}
        
        # 1. Load Model
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model not found at: {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        
        # 2. Batch Process ALL Avatars
        self._cache_all_avatars()

    def _load_model(self, path):
        print(f"📦 Loading Wav2Lip Model...")
        model = Wav2Lip()
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        s = checkpoint["state_dict"]
        new_s = {k.replace('module.', ''): v for k, v in s.items()}
        model.load_state_dict(new_s)
        return model.to(self.device).eval()

    def _cache_all_avatars(self):
        print(f"📂 Scanning for avatars in: {self.avatars_dir}")
        valid_exts = ['.jpg', '.jpeg', '.png']
        if not os.path.exists(self.avatars_dir):
            os.makedirs(self.avatars_dir)
            
        files = [f for f in os.listdir(self.avatars_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        
        if not files:
            print("⚠️ WARNING: No avatar images found! Please add images to backend/avatars/")
            return

        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                flip_input=False, device=self.device)

        for filename in files:
            path = os.path.join(self.avatars_dir, filename)
            print(f"   👤 Processing: {filename}...", end="")
            
            try:
                frame = cv2.imread(path)
                if frame is None: continue

                batch_det = detector.get_detections_for_batch(np.array([frame]))
                rect = batch_det[0]
                
                if rect is None:
                    print(" ❌ No face detected. Skipping.")
                    continue

                y1, y2, x1, x2 = rect[1], rect[3], rect[0], rect[2]
                pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
                y1 = max(0, y1 - pady1)
                y2 = min(frame.shape[0], y2 + pady2)
                x1 = max(0, x1 - padx1)
                x2 = min(frame.shape[1], x2 + padx2)
                
                cropped_face = frame[y1:y2, x1:x2]
                coords = (y1, y2, x1, x2)

                processed_tensor = self._prepare_face_tensor(cropped_face)
                
                # Pre-calculate blending mask for this avatar
                blend_mask = self._generate_blend_mask(y2-y1, x2-x1)

                self.avatar_cache[filename] = {
                    "full_frame": frame,
                    "coords": coords,
                    "tensor": processed_tensor,
                    "blend_mask": blend_mask
                }
                print(" ✅ Cached!")
            except Exception as e:
                print(f" ❌ Error: {e}")

    def _prepare_face_tensor(self, face_img):
        # Resize to 96x96 for Wav2Lip
        face_img = cv2.resize(face_img, (self.img_size, self.img_size))
        
        # [CRITICAL FIX] Mask the LOWER HALF (Mouth), not the Right Half
        img_masked = face_img.copy()
        img_masked[self.img_size//2:, :] = 0  # <--- FIXED LINE (Rows 48-96, All Cols)
        
        img_stacked = np.concatenate((img_masked, face_img), axis=2)
        img_tensor = torch.FloatTensor(img_stacked).permute(2, 0, 1) / 255.
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(self.device)

    def _generate_blend_mask(self, h, w):
        """Creates a soft feather mask to hide the square box edges"""
        mask = np.zeros((h, w), dtype=np.float32)
        pad_h, pad_w = int(h * 0.1), int(w * 0.1) 
        cv2.rectangle(mask, (pad_w, pad_h), (w - pad_w, h - pad_h), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 10)
        return np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    def infer(self, audio_path, output_path, avatar_filename):
        if avatar_filename not in self.avatar_cache:
            avatar_filename = list(self.avatar_cache.keys())[0]

        data = self.avatar_cache[avatar_filename]
        cached_tensor = data["tensor"]
        full_frame = data["full_frame"]
        blend_mask = data["blend_mask"]
        y1, y2, x1, x2 = data["coords"]
        
        # 1. Audio
        if not audio_path.endswith('.wav'):
            temp_wav = audio_path.replace('.mp3', '_temp.wav')
            subprocess.call(f'ffmpeg -y -i "{audio_path}" -strict -2 "{temp_wav}" -loglevel error', shell=True)
            audio_path = temp_wav

        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        
        # 2. Batching
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

        if not mel_chunks:
            print("⚠️ Audio too short, skipping video generation.")
            return

        mel_batches = [mel_chunks[i:i + self.batch_size] for i in range(0, len(mel_chunks), self.batch_size)]
        
        # 3. Direct FFmpeg Pipe (MUCH FASTER + BROWSER FRIENDLY)
        h, w = full_frame.shape[:2]
        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', '-i', audio_path, 
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', # Fix for odd dimensions
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', 
            '-crf', '25', '-preset', 'ultrafast',
            output_path
        ]
        try:
            # [CRITICAL FIX] Avoid stderr=PIPE to prevent deadlock
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=None)
        except Exception as e:
            print(f"❌ FFmpeg failed to start: {e}")
            raise
        
        # 4. Optimized Inference Loop
        original_face_patch = full_frame[y1:y2, x1:x2].astype(np.float32)
        bg_contribution = original_face_patch * (1.0 - blend_mask)
        canvas = full_frame.copy() # Reuse one canvas
        
        frame_count = 0
        total_frames = len(mel_chunks)
        print(f"   🎬 Generated {total_frames} audio frames. Starting video generation...")

        # 4. Inference
        for batch_mels in mel_batches:
            mels_tensor = torch.FloatTensor(np.array(batch_mels)).unsqueeze(1).to(self.device)
            curr_batch_size = len(batch_mels)
            faces_tensor = cached_tensor.expand(curr_batch_size, -1, -1, -1)
            
            with torch.no_grad():
                pred = self.model(mels_tensor, faces_tensor)
            
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p in pred:
                p_resized = cv2.resize(p, (x2 - x1, y2 - y1))
                blended_face = p_resized * blend_mask + bg_contribution
                canvas[y1:y2, x1:x2] = blended_face.astype(np.uint8)
                try:
                    process.stdin.write(canvas.tobytes())
                    frame_count += 1
                    if frame_count % 50 == 0:
                        print(f"      🎞️  Encoding: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end="\r")
                except BrokenPipeError:
                    print("❌ FFmpeg Pipeline CRASHED. Check console above for error.")
                    raise RuntimeError("FFmpeg Pipeline CRASHED.")
                
        process.stdin.close()
        process.wait()

        # Clean up temp audio if created
        if '_temp.wav' in audio_path:
            try: os.remove(audio_path)
            except: pass