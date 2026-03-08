"""
================================================================================
PHOTOGRAPHY CULLING & BENCHMARKING TOOL (v1.0)
================================================================================
Author: Ramazan Ertuğrul Aydoğan
Dependencies: ultralytics, rawpy, opencv-python, numpy, pillow, tqdm, torch

INSTALLATION:
  pip install ultralytics rawpy opencv-python numpy pillow tqdm torch torchvision

DESCRIPTION:
  1. AI SORTER:
     - Uses YOLOv8n to detect Humans and Animals.
     - Uses Subject-Specific Bounding Box Sharpness for Portraits/Wildlife.
     - Uses 8x8 Grid Analysis to determine Landscape Sharpness (ignoring Bokeh).
     - Uses 95th Percentile Highlight detection for correct Exposure.
     - Supports .RAF (Fuji), .JPG, .PNG.
     - Fast Embedded JPEG extraction for RAW files (No slow demosaicing).
     - SQLite Database to save progress (Resume capability).

  2. BENCHMARK GENERATOR:
     - Creates 3x3 high-res contact sheets (2400px) of sorted images.
     - Useful for quickly verifying if the AI made the right choices.

USAGE:
  Run the script in a terminal/PowerShell:
  python ai_cull_tool.py
================================================================================
"""

import os
import sys
import shutil
import sqlite3
import io
import warnings
from pathlib import Path

# External Libraries
import cv2
import numpy as np
import rawpy
from PIL import Image
from tqdm import tqdm
import torch

# Suppress YOLO warnings for a cleaner CLI
warnings.filterwarnings("ignore")
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "Smart_Sort_v1"
DB_NAME = "sort_progress.db"

# --- Thresholds ---
# Grid Sharpness: If any 1/8th tile is > 85, image is Sharp (Preserves Bokeh)
TILE_SHARPNESS_THRESHOLD = 85.0  

# Exposure: If top 5% highlights are > 50 brightness, keep image (Preserves Sunsets)
MIN_HIGHLIGHT_BRIGHTNESS = 50 

# Grid Settings for Benchmark
BENCH_GRID_DIM = 3        # 3x3
BENCH_TILE_SIZE = 800     # 800px per tile
# =================================================

class ImageUtils:
    """Helper class for image operations to ensure consistency."""
    
    @staticmethod
    def get_fast_image(filepath):
        """
        Reads image. For RAW files, extracts embedded JPEG for speed.
        Returns CV2 BGR image (numpy array) or None.
        """
        try:
            path_str = str(filepath)
            ext = filepath.suffix.lower()
            # List of RAW formats that need rawpy
            raw_exts = {'.raf', '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2'}
            
            if ext in raw_exts:
                with rawpy.imread(path_str) as raw:
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            img_pil = Image.open(io.BytesIO(thumb.data))
                            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                        else:
                            # Fallback if no embedded JPEG is found (slower but works)
                            return cv2.cvtColor(raw.postprocess(), cv2.COLOR_RGB2BGR)
                    except Exception:
                        return None
            else:
                return cv2.imread(path_str)
        except Exception as e:
            return None

    @staticmethod   # <--- THIS WAS MISSING IN YOUR VERSION
    def get_tiled_sharpness(img_gray):
        """
        Splits image into 8x8 grid. Returns the maximum sharpness found in any tile.
        This allows a subject to be sharp while the background is blurry (Bokeh).
        """
        h, w = img_gray.shape
        h_step = max(1, h // 8)
        w_step = max(1, w // 8)
        
        max_sharpness = 0.0
        
        for y in range(0, h - h_step + 1, h_step):
            for x in range(0, w - w_step + 1, w_step):
                tile = img_gray[y:y+h_step, x:x+w_step]
                val = cv2.Laplacian(tile, cv2.CV_64F).var()
                if val > max_sharpness:
                    max_sharpness = val
                    
        return max_sharpness

class Benchmarker:
    """Handles the creation of contact sheets."""
    def __init__(self):
        self.canvas_w = BENCH_TILE_SIZE * BENCH_GRID_DIM
        self.canvas_h = BENCH_TILE_SIZE * BENCH_GRID_DIM

    def create_tile(self, filepath):
        img = ImageUtils.get_fast_image(filepath)
        if img is None: return None

        h, w = img.shape[:2]
        scale = BENCH_TILE_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        tile = np.zeros((BENCH_TILE_SIZE, BENCH_TILE_SIZE, 3), dtype=np.uint8)
        
        y_off = (BENCH_TILE_SIZE - new_h) // 2
        x_off = (BENCH_TILE_SIZE - new_w) // 2
        tile[y_off:y_off+new_h, x_off:x_off+new_w] = img_resized

        text = filepath.name
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Drop Shadow
        cv2.putText(tile, text, (15, BENCH_TILE_SIZE - 15), font, 1.2, (0,0,0), 5)
        # White Text
        cv2.putText(tile, text, (15, BENCH_TILE_SIZE - 15), font, 1.2, (255,255,255), 2)
        return tile

    def process_directory(self, target_root):
        """Scans the Output folder for subfolders and makes grids."""
        target_path = Path(target_root) / OUTPUT_FOLDER_NAME
        if not target_path.exists():
            print(f"[!] Output folder '{OUTPUT_FOLDER_NAME}' not found in {target_root}")
            return

        valid_exts = {'.jpg', '.jpeg', '.raf', '.png'}
        folders_to_scan =[]
        
        for root, dirs, files in os.walk(target_path):
            if any(f.lower().endswith(tuple(valid_exts)) for f in files):
                folders_to_scan.append(Path(root))

        if not folders_to_scan:
            print("[!] No images found to benchmark.")
            return

        print(f"\n[BENCHMARK] Generating grids for {len(folders_to_scan)} folders inside {target_root.name}...")

        for folder in tqdm(folders_to_scan, unit="dir"):
            images = sorted([
                p for p in folder.glob('*') 
                if p.suffix.lower() in valid_exts 
                and not p.name.startswith('_BENCHMARK')
            ])
            
            if not images: continue

            chunk_size = BENCH_GRID_DIM * BENCH_GRID_DIM
            batches = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]

            for i, batch in enumerate(batches):
                full_grid = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
                
                for idx, img_path in enumerate(batch):
                    tile = self.create_tile(img_path)
                    if tile is not None:
                        row = idx // BENCH_GRID_DIM
                        col = idx % BENCH_GRID_DIM
                        y = row * BENCH_TILE_SIZE
                        x = col * BENCH_TILE_SIZE
                        full_grid[y:y+BENCH_TILE_SIZE, x:x+BENCH_TILE_SIZE] = tile
                
                out_name = f"_BENCHMARK_Part_{i+1}.jpg"
                cv2.imwrite(str(folder / out_name), full_grid)

class AISorter:
    """Handles logic for categorization, moving files, and database."""
    def __init__(self):
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.model = None # Lazy load
        self.valid_exts = {'.jpg', '.jpeg', '.png', '.raf', '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2'}
        self.animal_classes = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23} # COCO dataset IDs

    def load_model(self):
        if self.model is None:
            print(f"Loading YOLOv8n on {'GPU (CUDA)' if self.device == '0' else 'CPU'}...")
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')

    def setup_db(self, root_dest):
        dest_path = Path(root_dest) / OUTPUT_FOLDER_NAME
        dest_path.mkdir(parents=True, exist_ok=True)
        db_path = dest_path / DB_NAME
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS processed (filepath TEXT PRIMARY KEY, category TEXT)")
        conn.commit()
        return conn, cursor

    def analyze(self, img):
        """Returns category string based on AI, Sharpness, and Exposure."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. EXPOSURE CHECK
        highlight_val = np.percentile(gray, 95)
        is_too_dark = highlight_val < MIN_HIGHLIGHT_BRIGHTNESS
        
        # 2. GLOBAL SHARPNESS CHECK (For Landscapes)
        peak_global_sharpness = ImageUtils.get_tiled_sharpness(gray)
        is_globally_sharp = peak_global_sharpness > TILE_SHARPNESS_THRESHOLD

        # 3. AI DETECTION
        scale = 1.0
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            detect_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        else:
            detect_img = img

        results = self.model(detect_img, device=self.device, verbose=False)[0]
        
        has_human = False
        has_animal = False
        max_subject_sharpness = 0.0
        scale_back = 1.0 / scale
        
        # Calculate SUBJECT-SPECIFIC sharpness (The Donut-Hole fix)
        for box in results.boxes:
            cls = int(box.cls[0])
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            
            # Map box back to original full-res image boundaries securely
            fx1, fy1 = max(0, int(bx1 * scale_back)), max(0, int(by1 * scale_back))
            fx2, fy2 = min(w, int(bx2 * scale_back)), min(h, int(by2 * scale_back))
            
            crop = gray[fy1:fy2, fx1:fx2]
            
            if crop.size > 0:
                score = cv2.Laplacian(crop, cv2.CV_64F).var()
                max_subject_sharpness = max(max_subject_sharpness, score)
                
            if cls == 0: has_human = True
            elif cls in self.animal_classes: has_animal = True
            
        is_subject_sharp = max_subject_sharpness > TILE_SHARPNESS_THRESHOLD

        # 4. DECISION LOGIC
        if has_human:
            if is_subject_sharp: return "Human/Sharp_Portrait"
            elif not is_too_dark: return "Human/Artistic_Dark"
            else: return "Human/Blurry_Discard"
            
        elif has_animal:
            if is_subject_sharp: return "Animal/Sharp"
            else: return "Animal/Blurry_Discard"
            
        else: # Landscape / Object
            if is_too_dark: return "Landscape/Blurry_Discard"
            elif is_globally_sharp: return "Landscape/Sharp_Detail"
            elif highlight_val > 150: return "Landscape/Artistic_LowLight"
            else: return "Landscape/Blurry_Discard"

    def flatten_directory(self, root_path):
        """Moves files from subfolders back to root and cleans up."""
        root = Path(root_path)
        output_dir = root / OUTPUT_FOLDER_NAME
        
        if not output_dir.exists():
            return
            
        print(f"\n[RESET] Moving files back to root: {root}")
        moved_count = 0

        # Move files out of the output folder
        for p in output_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in self.valid_exts:
                dest = root / p.name
                counter = 1
                while dest.exists():
                    dest = root / f"{p.stem}_{counter}{p.suffix}"
                    counter += 1
                
                shutil.move(str(p), str(dest))
                moved_count += 1

        # Delete Output Folder and DB
        try:
            shutil.rmtree(output_dir)
            print("[RESET] Deleted old output folder and database.")
        except Exception as e:
            print(f"[RESET] Warning: Could not delete output folder completely: {e}")

        print(f"[RESET] Restored {moved_count} images.\n")

    def run_sort(self, root_path, reset_mode=False):
        root = Path(root_path)
        if not root.exists():
            print(f"[!] Path does not exist: {root}")
            return

        if reset_mode:
            self.flatten_directory(root)
        
        self.load_model()
        conn, cursor = self.setup_db(root)
        
        # Gather Files (Exclude the Output folder to prevent infinite loops)
        all_files =[]
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in self.valid_exts:
                if OUTPUT_FOLDER_NAME not in p.parts:
                    all_files.append(p)
        
        print(f"\nTarget: {root}")
        print(f"Found {len(all_files)} images to process.")
        
        if len(all_files) == 0:
            return

        pbar = tqdm(all_files, unit="img")
        for file_path in pbar:
            # Check DB
            cursor.execute("SELECT category FROM processed WHERE filepath = ?", (str(file_path.name),))
            if cursor.fetchone():
                continue

            try:
                img = ImageUtils.get_fast_image(file_path)
                if img is None:
                    cat = "Failed_Read"
                else:
                    cat = self.analyze(img)

                dest_dir = root / OUTPUT_FOLDER_NAME / cat
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                final_path = dest_dir / file_path.name
                counter = 1
                while final_path.exists():
                    final_path = dest_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
                    counter += 1

                shutil.move(str(file_path), str(final_path))
                
                cursor.execute("INSERT INTO processed (filepath, category) VALUES (?, ?)", 
                               (str(file_path.name), cat))
                conn.commit()
                
                pbar.set_description(f"Processing: {file_path.name[:15]:<15} | Cat: {cat.split('/')[-1][:18]:<18}")

            except Exception as e:
                print(f"\n[!] Error processing {file_path.name}: {e}")
        
        conn.close()
        
        # Final Cleanup of empty directories in root
        for p in root.rglob('*'):
            if p.is_dir() and OUTPUT_FOLDER_NAME not in p.parts and not any(p.iterdir()):
                try: p.rmdir()
                except: pass

        print(f"\n[DONE] Sorting complete for: {root.name}")


# ================= MENU SYSTEM =================

def get_input_paths():
    paths =[]
    print("\n--- ENTER FOLDER PATHS ---")
    print("Type absolute paths (e.g., C:/Photos/Trip).")
    print("Type 'end' when finished.")
    
    while True:
        p = input("Path > ").strip().strip('"').strip("'")
        if p.lower() == 'end':
            break
        if os.path.isdir(p):
            paths.append(Path(p))
            print(f"  [+] Added: {p}")
        else:
            print("  [!] Invalid folder path. Try again.")
    return paths

def main():
    print("="*60)
    print("       AI PHOTOGRAPHY CULLING SUITE (v5.0)       ")
    print("       Sorts by Sharpness, Exposure & Content    ")
    print("="*60)

    print("\n[1] AI Image Sorting (Categorize)")
    print("[2] Benchmark Grids (Create Contact Sheets)")
    choice_task = input("\nSelect Task (1 or 2): ").strip()

    if choice_task not in ['1', '2']:
        print("Invalid choice. Exiting.")
        return

    print("\n--- SELECT SOURCE ---")
    print("[1] Current Folder (Where this script is running)")
    print("[2] Custom Folder Paths")
    choice_source = input("Select Source (1 or 2): ").strip()

    target_paths =[]
    if choice_source == '1':
        target_paths = [Path(os.getcwd())]
    elif choice_source == '2':
        target_paths = get_input_paths()
    else:
        print("Invalid choice.")
        return

    if not target_paths:
        print("No paths selected. Exiting.")
        return

    if choice_task == '1':
        sorter = AISorter()
        print("\n--- SORTING MODE ---")
        print("[1] Continue/Resume (Skip already processed files)")
        print("[2] RESET & START FRESH (Move files back to root, clear DB)")
        choice_mode = input("Select Mode (1 or 2): ").strip()
        
        reset = (choice_mode == '2')
        for path in target_paths:
            sorter.run_sort(path, reset_mode=reset)

    elif choice_task == '2':
        bencher = Benchmarker()
        print("\n--- BENCHMARK MODE ---")
        for path in target_paths:
            bencher.process_directory(path)

    print("\n" + "="*60)
    print("All operations completed.")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Operation stopped by user.")
        sys.exit()
