#!/usr/bin/env python3
"""
Quick Start Example
Demonstrates how to use the AI Vision Benchmark
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from models.commercial.gpt4o import GPT4OModel
    from utils.scorer import BenchmarkScorer
    from utils.reporter import ResultsReporter
    import yaml
    
    def quick_demo():
        """Run a quick demo of the benchmark system"""
        print("🧪 AI Vision Benchmark - Quick Start Demo")
        print("=" * 50)
        
        # Check if config exists
        config_path = project_root / "config" / "config.yaml"
        if not config_path.exists():
            print("❌ Configuration file not found!")
            print("Please run: python scripts/setup_environment.py")
            return
            
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Check if OpenAI API key is configured
        openai_key = config.get('openai', {}).get('api_key', '')
        if not openai_key or openai_key == "your-openai-api-key-here":
            print("❌ OpenAI API key not configured!")
            print("Please edit config/config.yaml and add your OpenAI API key")
            return
            
        print("✅ Configuration loaded successfully")
        
        # Initialize components
        try:
            gpt4o_model = GPT4OModel(config['openai'])
            scorer = BenchmarkScorer(config['scoring'])
            reporter = ResultsReporter(config['output'])
            print("✅ Components initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            return
            
        # Check for test images
        test_images_dir = project_root / "data" / "images"
        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        
        if not image_files:
            print("❌ No test images found!")
            print(f"Please add images to: {test_images_dir}")
            return
            
        print(f"✅ Found {len(image_files)} test image(s)")
        
        # Run a simple test
        test_image = image_files[0]
        test_prompt = "Describe what you see in this image"
        
        print(f"\n🔍 Testing with image: {test_image.name}")
        print(f"📝 Prompt: {test_prompt}")
        
        try:
            # Run detection
            result = gpt4o_model.detect_objects(str(test_image), test_prompt)
            
            # Score the result
            score = scorer.score_card_detection(result, test_prompt)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Detections found: {len(result['detections'])}")
            print(f"   Overall score: {score.get('Overall', 0):.2f}/5")
            
            if result['response_text']:
                print(f"\n💬 Response preview:")
                preview = result['response_text'][:200] + "..." if len(result['response_text']) > 200 else result['response_text']
                print(f"   {preview}")
                
            print(f"\n✅ Test completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during test: {e}")
            return
            
        print("\n" + "=" * 50)
        print("🎉 Quick demo completed!")
        print("\nNext steps:")
        print("1. Add more test images to data/images/")
        print("2. Add test videos to data/videos/")
        print("3. Configure other API keys in config.yaml")
        print("4. Run full benchmark: python run_benchmark.py")
        
    if __name__ == "__main__":
        quick_demo()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt") 