# 🧪 AI Object & Person Detection Benchmark

A comprehensive benchmarking project comparing commercial and open-source AI models for real-world object detection and person re-identification tasks.

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone <your-repo>
   cd VisionAI
   ```

2. **Install Dependencies**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install requirements
   pip install -r requirements.txt
   ```

3. **Configure API Keys**
   ```bash
   cp config/config.template.yaml config/config.yaml
   # Edit config.yaml with your API keys
   ```

4. **Add Test Data**
   - Place test images in `data/images/`
   - Place test videos in `data/videos/`

## 📁 Project Structure

```
VisionAI/
├── config/                 # Configuration files
├── data/                   # Test data
│   ├── images/            # Test images
│   └── videos/            # Test videos
├── models/                # Model implementations
│   ├── commercial/        # Commercial API integrations
│   └── opensource/        # Open-source model implementations
├── results/               # Benchmark results
├── scripts/               # Utility scripts
├── tests/                 # Test cases
└── docs/                  # Documentation
```

## 🔧 Models Tested

- **Commercial**: OpenAI GPT-4o Vision, Google Gemini Pro Vision, AWS Rekognition
- **Open Source**: YOLOv8 + GroundingDINO, Segment Anything + BLIP-2

## 📊 Tests

1. **Test 1**: Greeting Card Wall (Image Detection)
2. **Test 2**: Person Re-Identification (Video Tracking)

## 🏃‍♂️ Running Benchmarks

```bash
# Run all benchmarks
python run_benchmark.py

# Run specific test
python run_benchmark.py --test card_detection
python run_benchmark.py --test person_reid

# Run specific model
python run_benchmark.py --model gpt4o
```

See `project.md` for detailed project specifications and scoring rubrics. 