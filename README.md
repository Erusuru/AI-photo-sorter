# AI Photography Culling Suite (v1.0)

A testing stage Python tool for photographers to automate the culling process. It sorts images based on **Subject Sharpness**, **Exposure**, and **Content** (Humans vs Animals vs Landscapes) using YOLOv8 and Grid-based analysis.

## Features
- **Smart Sharpness:** Ignores "Bokeh" (blurry backgrounds) and focuses on the subject.
- **Smart Exposure:** Uses 95th Percentile Highlight detection to preserve dark artistic shots (silhouettes).
- **Format Support:** Supports `.RAF` (Fuji Raw), `.JPG`, `.PNG`. fast-extracts embedded JPEGs from RAWs for speed.
- **Benchmarking:** Generates 3x3 high-res contact sheets for easy verification.
- **Resume Capability:** Tracks progress via SQLite, allowing you to stop and resume later.

## Installation

1. Install Python.
2. Run: `pip install -r requirements.txt`
3. Run the tool: `python imagesort_fast.py`