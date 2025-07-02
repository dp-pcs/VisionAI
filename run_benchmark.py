#!/usr/bin/env python3
"""
AI Vision Benchmark Runner
Main script to execute benchmarks across different AI models
"""

import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json

from models.commercial.gpt4o import GPT4OModel
from models.commercial.gemini import GeminiModel
from models.commercial.aws_rekognition import AWSRekognitionModel
from models.opensource.yolo_grounding_dino import YOLOGroundingDINOModel
from models.opensource.sam_blip2 import SAMBlip2Model
from utils.scorer import BenchmarkScorer
from utils.reporter import ResultsReporter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Main benchmark runner class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the benchmark runner"""
        self.config = self._load_config(config_path)
        self.models = self._initialize_models()
        self.scorer = BenchmarkScorer(self.config['scoring'])
        self.reporter = ResultsReporter(self.config['output'])
        
        # Create results directory
        results_dir = Path(self.config['output']['results_dir'])
        results_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            logger.error("Please copy config.template.yaml to config.yaml and configure your API keys")
            raise
            
    def _initialize_models(self) -> Dict:
        """Initialize all enabled models"""
        models = {}
        
        model_config = self.config['models']
        
        # Initialize commercial models
        if model_config['commercial'].get('gpt4o'):
            models['gpt4o'] = GPT4OModel(self.config['openai'])
            
        if model_config['commercial'].get('gemini'):
            models['gemini'] = GeminiModel(self.config['google'])
            
        if model_config['commercial'].get('aws_rekognition'):
            models['aws_rekognition'] = AWSRekognitionModel(self.config['aws'])
            
        # Initialize open source models
        if model_config['opensource'].get('yolov8_grounding_dino'):
            models['yolov8_grounding_dino'] = YOLOGroundingDINOModel()
            
        if model_config['opensource'].get('segment_anything_blip2'):
            models['sam_blip2'] = SAMBlip2Model()
            
        logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
        return models
        
    def run_card_detection_test(self, models_to_test: List[str] = None) -> Dict:
        """Run the greeting card detection test"""
        logger.info("Running Card Detection Test")
        
        test_config = self.config['tests']['card_detection']
        if not test_config['enabled']:
            logger.info("Card detection test is disabled")
            return {}
            
        results = {}
        models_to_run = models_to_test or list(self.models.keys())
        
        for model_name in models_to_run:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found, skipping")
                continue
                
            logger.info(f"Testing {model_name} on card detection")
            model = self.models[model_name]
            model_results = []
            
            for image_path in test_config['test_images']:
                full_image_path = Path("data/images") / image_path
                
                if not full_image_path.exists():
                    logger.warning(f"Test image {full_image_path} not found")
                    continue
                    
                for prompt in test_config['prompts']:
                    logger.info(f"Running prompt: {prompt}")
                    
                    try:
                        result = model.detect_objects(str(full_image_path), prompt)
                        score = self.scorer.score_card_detection(result, prompt)
                        
                        model_results.append({
                            'image': image_path,
                            'prompt': prompt,
                            'result': result,
                            'score': score,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        logger.error(f"Error running {model_name} on {prompt}: {e}")
                        model_results.append({
                            'image': image_path,
                            'prompt': prompt,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
                        
            results[model_name] = model_results
            
        return results
        
    def run_person_reid_test(self, models_to_test: List[str] = None) -> Dict:
        """Run the person re-identification test"""
        logger.info("Running Person Re-ID Test")
        
        test_config = self.config['tests']['person_reid']
        if not test_config['enabled']:
            logger.info("Person re-ID test is disabled")
            return {}
            
        results = {}
        models_to_run = models_to_test or list(self.models.keys())
        
        for model_name in models_to_run:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found, skipping")
                continue
                
            logger.info(f"Testing {model_name} on person re-ID")
            model = self.models[model_name]
            model_results = []
            
            for video_path in test_config['test_videos']:
                full_video_path = Path("data/videos") / video_path
                
                if not full_video_path.exists():
                    logger.warning(f"Test video {full_video_path} not found")
                    continue
                    
                for prompt in test_config['prompts']:
                    logger.info(f"Running prompt: {prompt}")
                    
                    try:
                        result = model.track_person(str(full_video_path), prompt)
                        score = self.scorer.score_person_reid(result, prompt)
                        
                        model_results.append({
                            'video': video_path,
                            'prompt': prompt,
                            'result': result,
                            'score': score,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        logger.error(f"Error running {model_name} on {prompt}: {e}")
                        model_results.append({
                            'video': video_path,
                            'prompt': prompt,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
                        
            results[model_name] = model_results
            
        return results
        
    def run_all_benchmarks(self, models_to_test: List[str] = None) -> Dict:
        """Run all enabled benchmarks"""
        logger.info("Starting benchmark run")
        
        all_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models_tested': models_to_test or list(self.models.keys()),
                'config': self.config
            },
            'card_detection': {},
            'person_reid': {}
        }
        
        # Run card detection test
        if self.config['tests']['card_detection']['enabled']:
            all_results['card_detection'] = self.run_card_detection_test(models_to_test)
            
        # Run person re-ID test  
        if self.config['tests']['person_reid']['enabled']:
            all_results['person_reid'] = self.run_person_reid_test(models_to_test)
            
        # Save results
        self.reporter.save_results(all_results)
        
        logger.info("Benchmark run completed")
        return all_results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Vision Benchmark Runner")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--test", choices=['card_detection', 'person_reid'], help="Run specific test only")
    parser.add_argument("--model", help="Run specific model only")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    try:
        runner = BenchmarkRunner(args.config)
        
        if args.list_models:
            print("Available models:")
            for model_name in runner.models.keys():
                print(f"  - {model_name}")
            return
            
        models_to_test = [args.model] if args.model else None
        
        if args.test == 'card_detection':
            results = runner.run_card_detection_test(models_to_test)
        elif args.test == 'person_reid':
            results = runner.run_person_reid_test(models_to_test)
        else:
            results = runner.run_all_benchmarks(models_to_test)
            
        print("Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main() 