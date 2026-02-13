from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos')

parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--face', type=str, required=True)
parser.add_argument('--audio', type=str, required=True)
parser.add_argument('--outfile', type=str, default='results/result_voice.mp4')
parser.add_argument('--static', type=bool, default=False)
parser.add_argument('--fps', type=float, default=25.)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0])
parser.add_argument('--face_det_batch_size', type=int, default=16)
parser.add_argument('--wav2lip_batch_size', type=int, default=128)
parser.add_argument('--resize_factor', default=1, type=int)
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1])
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1])
parser.add_argument('--rotate', default=False, action='store_true')
parser.add_argument('--nosmooth', default=False, action='store_true')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes): window = boxes[len(boxes) - T:]
		else: window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
	batch_size = args.face_det_batch_size
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: raise RuntimeError('Image too big for GPU.')
			batch_size //= 2
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None: raise ValueError('Face not detected!')
		y1, y2, x1, x2 = max(0, rect[1] - pady1), min(image.shape[0], rect[3] + pady2), max(0, rect[0] - padx1), min(image.shape[1], rect[2] + padx2)
		results.append([x1, y1, x2, y2])
	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	return [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
	face_det_results = face_detect(frames) if not args.static else face_detect([frames[0]])
	for i, m in enumerate(mels):
		idx = 0 if args.static else i % len(frames)
		frame_to_save, (face, coords) = frames[idx].copy(), face_det_results[idx]
		face = cv2.resize(face, (args.img_size, args.img_size))
		img_batch.append(face); mel_batch.append(m); frame_batch.append(frame_to_save); coords_batch.append(coords)
		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
			img_masked = img_batch.copy(); img_masked[:, args.img_size//2:] = 0
			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
		img_masked = img_batch.copy(); img_masked[:, args.img_size//2:] = 0
		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
		yield img_batch, mel_batch, frame_batch, coords_batch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_model(path):
	model = Wav2Lip()
	checkpoint = torch.load(path) if device == 'cuda' else torch.load(path, map_location=lambda s, l: s)
	s = checkpoint["state_dict"]
	new_s = {k.replace('module.', ''): v for k, v in s.items()}
	model.load_state_dict(new_s)
	return model.to(device).eval()

def main():
	script_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
	temp_dir = f"{script_dir}/temp"
	if not os.path.exists(temp_dir): os.makedirs(temp_dir)

	if args.static:
		full_frames = [cv2.imread(args.face)]; fps = args.fps
	else:
		video_stream = cv2.VideoCapture(args.face); fps = video_stream.get(cv2.CAP_PROP_FPS)
		full_frames = []
		while True:
			ret, frame = video_stream.read()
			if not ret: break
			full_frames.append(frame)
		video_stream.release()

	temp_wav = f"{temp_dir}/temp.wav"
	subprocess.call(f'ffmpeg -y -i "{args.audio}" -strict -2 "{temp_wav}"', shell=True)
	
	wav = audio.load_wav(temp_wav, 16000); mel = audio.melspectrogram(wav)
	mel_chunks = []
	for i in range(0, 100000):
		start_idx = int(i * (80./fps))
		if start_idx + 16 > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - 16:]); break
		mel_chunks.append(mel[:, start_idx : start_idx + 16])

	full_frames = full_frames[:len(mel_chunks)]
	model = load_model(args.checkpoint_path)
	result_avi = f"{temp_dir}/result.avi"
	out = cv2.VideoWriter(result_avi, cv2.VideoWriter_fourcc(*'DIVX'), fps, (full_frames[0].shape[1], full_frames[0].shape[0]))

	for img_batch, mel_batch, frames, coords in tqdm(datagen(full_frames, mel_chunks)):
		img_tensor = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_tensor = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
		with torch.no_grad(): pred = model(mel_tensor, img_tensor)
		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			f[y1:y2, x1:x2] = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1)); out.write(f)
	out.release()
	subprocess.call(f'ffmpeg -y -i "{temp_wav}" -i "{result_avi}" -strict -2 -q:v 1 "{args.outfile}"', shell=True)

if __name__ == '__main__': main()