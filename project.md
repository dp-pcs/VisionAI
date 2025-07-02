
# 🧪 AI Object & Person Detection Benchmark

## 📘 Overview
This project benchmarks real-world object and person detection across commercial and open-source AI models using structured prompts, consistent inputs, and rubric-based scoring. It includes two primary tests—image-based card detection and video-based person re-identification.

---

## 📋 Vendors/Solutions Compared

| Type          | Model(s)                         | Interface           |
|---------------|----------------------------------|---------------------|
| Commercial    | OpenAI GPT-4o Vision             | ChatGPT / API       |
|               | Google Gemini Pro Vision         | Web / API           |
|               | AWS Rekognition (Image + Video)  | Console / API       |
| Open Source   | YOLOv8 + GroundingDINO           | Python, CLI         |
|               | Segment Anything + BLIP-2        | Python, CLI         |

---

## 🖼️ Test 1: Greeting Card Wall (Image)

### 🎯 Goal
Evaluate each model's ability to:
- Detect section headers (e.g., “Romantic Birthday”)
- Identify illustrations on cards (e.g., bear, dog, balloons)
- Parse text on card fronts (e.g., “Happy Birthday”)

### 📥 Input
- High-resolution image of a greeting card wall

### 💬 Prompts
- “Where are the ‘Romantic Birthday’ cards?”
- “Show me all cards with a bear on them.”
- “Which cards have the phrase ‘Happy Birthday’ visible on the front?”

### 📈 Scoring Rubric

| Metric                    | Description                                             | Score (0–5) |
|---------------------------|---------------------------------------------------------|-------------|
| Text-Region Localization  | Finds section headers like “Romantic Birthday”          |             |
| Object-Level Detection    | Finds cards with requested illustrations                |             |
| OCR + Semantic Parsing    | Parses and understands visible text                     |             |
| Visual Bounding Clarity   | Bounding box accuracy and completeness                  |             |
| Instruction Following     | Follows the prompt and matches requested criteria       |             |

---

## 🎥 Test 2: Person Re-Identification (Video)

### 🎯 Goal
Assess the ability to visually track a person in video based on description.

### 📥 Input
- 10–30 second video clip featuring multiple people, movement, and occlusion

### 💬 Prompts
- “Find the person in the red hoodie and black pants.”
- “Show me when [target] enters the frame and track them.”

### 📈 Scoring Rubric

| Metric                    | Description                                               | Score (0–5) |
|---------------------------|-----------------------------------------------------------|-------------|
| Target Re-ID Accuracy     | Correctly identifies the person and tracks them           |             |
| Robustness to Occlusion   | Maintains ID despite partial obstruction                  |             |
| Temporal Awareness        | Recognizes entry/exit in the video timeline               |             |
| Visual Reasoning          | Explains or justifies selection (if supported)            |             |
| Instruction Following     | Matches the prompt with high fidelity                     |             |

---

## 📤 Output Format

For each vendor and test:
- Screenshots or JSON of detected regions
- Written explanation or logs (if available)
- Completed rubric with 0–5 scores
- Optional: timing info for API-based solutions

---

## ✅ Project Checklist

### 📁 Setup
- [ ] Collect and organize image and video test inputs
- [ ] Preprocess media as needed (format, resolution)
- [ ] Ensure all test data is reproducible and documented

### 🔧 Tool Access
- [ ] OpenAI GPT-4o access via ChatGPT or API
- [ ] Gemini Pro Vision access (API or Bard)
- [ ] AWS Rekognition credentials (IAM + SDK)
- [ ] Python env for YOLOv8 + GroundingDINO
- [ ] Python env for Segment Anything + BLIP-2

### 🚀 Run Benchmarks
- [ ] Execute each model with consistent prompts
- [ ] Log all output (visual + text)
- [ ] Score based on rubric

### 🧾 Documentation
- [ ] Fill results table
- [ ] Add insights and reflections
- [ ] Publish results to GitHub/Notion
- [ ] Include in article writeup

---

## 🔗 References

- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [BLIP-2](https://github.com/salesforce/LAVIS)
- [AWS Rekognition Docs](https://docs.aws.amazon.com/rekognition/)
- [GPT-4o](https://openai.com/index/gpt-4o/)
- [Gemini](https://deepmind.google/technologies/gemini/)

---

*Created by David Proctor — July 2025*
