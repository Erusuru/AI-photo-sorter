# 📸 AI Photography Culling Suite V1.0

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Mac%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**A high-performance automated culling tool for photographers.**  
Sorts thousands of photos in minutes based on **Sharpness**, **Exposure**, and **Subject Content** (Humans vs. Animals vs. Landscapes) using YOLOv8 AI.

---

## 🚀 Features

### 🧠 Smart Analysis
*   **"Donut-Hole" Sharpness:** Unlike basic tools, this checks if the **Subject** is sharp while ignoring blurry backgrounds (Bokeh).
*   **Smart Exposure:** Uses 95th Percentile Highlight detection. It knows the difference between "Underexposed" and "Artistic Silhouette."
*   **Content Awareness:** Automatically separates Portraits, Wildlife, and Landscapes into their own folders.

### ⚡ Performance
*   **Universal RAW Support:** Zero-lag preview extraction for `.RAF`, `.CR3`, `.ARW`, `.NEF`, and more.
*   **Resume Capability:** Uses an SQLite database to track progress. Crashed? Stopped? It resumes exactly where it left off.
*   **Benchmark Mode:** Generates high-res 3x3 contact sheets to verify the AI's decisions visually.

---

## 📷 Supported Cameras
The tool uses `rawpy` to extract embedded previews, supporting virtually all major formats:

| Brand | Extensions |
| :--- | :--- |
| **Canon** | .CR2, .CR3 |
| **Sony** | .ARW |
| **Nikon** | .NEF |
| **Fujifilm** | .RAF |
| **Olympus** | .ORF |
| **Generic** | .DNG, .JPG, .PNG |

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Erusuru/AI-photo-sorter.git
   cd AI-photo-sorter
