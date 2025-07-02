"""
Results Reporter
Generates benchmark reports in various formats (JSON, CSV, HTML)
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ResultsReporter:
    """Generates and saves benchmark results in multiple formats"""
    
    def __init__(self, output_config: Dict[str, Any]):
        """Initialize reporter with output configuration"""
        self.results_dir = Path(output_config['results_dir'])
        self.save_images = output_config.get('save_images', True)
        self.save_videos = output_config.get('save_videos', True)
        self.export_formats = output_config.get('export_format', ['json'])
        self.logger = logging.getLogger(__name__)
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
    def save_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save benchmark results in configured formats
        
        Args:
            results: Complete benchmark results dictionary
            
        Returns:
            Dictionary mapping format names to file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        try:
            # Save in each requested format
            for format_name in self.export_formats:
                if format_name == 'json':
                    file_path = self._save_json(results, timestamp)
                    saved_files['json'] = str(file_path)
                    
                elif format_name == 'csv':
                    file_path = self._save_csv(results, timestamp)
                    saved_files['csv'] = str(file_path)
                    
                elif format_name == 'html':
                    file_path = self._save_html(results, timestamp)
                    saved_files['html'] = str(file_path)
                    
                else:
                    self.logger.warning(f"Unknown export format: {format_name}")
                    
            self.logger.info(f"Saved results in {len(saved_files)} formats")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return {}
            
    def _save_json(self, results: Dict[str, Any], timestamp: str) -> Path:
        """Save results as JSON"""
        filename = f"benchmark_results_{timestamp}.json"
        file_path = self.results_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Saved JSON results to {file_path}")
        return file_path
        
    def _save_csv(self, results: Dict[str, Any], timestamp: str) -> Path:
        """Save results as CSV"""
        filename = f"benchmark_results_{timestamp}.csv"
        file_path = self.results_dir / filename
        
        # Flatten results for CSV format
        csv_data = self._flatten_results_for_csv(results)
        
        if csv_data:
            with open(file_path, 'w', newline='') as f:
                fieldnames = csv_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
                
        self.logger.info(f"Saved CSV results to {file_path}")
        return file_path
        
    def _save_html(self, results: Dict[str, Any], timestamp: str) -> Path:
        """Save results as HTML report"""
        filename = f"benchmark_report_{timestamp}.html"
        file_path = self.results_dir / filename
        
        html_content = self._generate_html_report(results)
        
        with open(file_path, 'w') as f:
            f.write(html_content)
            
        self.logger.info(f"Saved HTML report to {file_path}")
        return file_path
        
    def _flatten_results_for_csv(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten nested results dictionary for CSV export"""
        csv_rows = []
        
        for test_type in ['card_detection', 'person_reid']:
            if test_type not in results:
                continue
                
            test_results = results[test_type]
            for model_name, model_results in test_results.items():
                if not model_results:  # Skip empty results
                    continue
                    
                for result in model_results:
                    row = {
                        'test_type': test_type,
                        'model': model_name,
                        'timestamp': result.get('timestamp', ''),
                        'image': result.get('image', ''),
                        'video': result.get('video', ''),
                        'prompt': result.get('prompt', ''),
                        'processing_time': result.get('result', {}).get('processing_time', ''),
                        'has_error': 'error' in result
                    }
                    
                    # Add score metrics
                    scores = result.get('score', {})
                    for metric, score in scores.items():
                        row[f'score_{metric.replace(" ", "_").lower()}'] = score
                        
                    # Add detection/tracking counts
                    result_data = result.get('result', {})
                    if 'detections' in result_data:
                        row['detection_count'] = len(result_data['detections'])
                    if 'tracks' in result_data:
                        row['track_count'] = len(result_data['tracks'])
                        
                    csv_rows.append(row)
                    
        return csv_rows
        
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report"""
        metadata = results.get('metadata', {})
        timestamp = metadata.get('timestamp', datetime.now().isoformat())
        models_tested = metadata.get('models_tested', [])
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Vision Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
        .test-section {{ margin-bottom: 40px; }}
        .model-results {{ margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .scores-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .scores-table th, .scores-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .scores-table th {{ background-color: #f2f2f2; }}
        .score-good {{ background-color: #d4edda; }}
        .score-medium {{ background-color: #fff3cd; }}
        .score-poor {{ background-color: #f8d7da; }}
        .summary {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 30px; }}
        h1, h2, h3 {{ color: #333; }}
        .error {{ color: #dc3545; font-style: italic; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 AI Vision Benchmark Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Models Tested:</strong> {', '.join(models_tested)}</p>
    </div>
"""
        
        # Generate sections for each test
        for test_type in ['card_detection', 'person_reid']:
            if test_type not in results or not results[test_type]:
                continue
                
            test_name = "Card Detection" if test_type == 'card_detection' else "Person Re-Identification"
            html += f'<div class="test-section"><h2>📊 {test_name} Results</h2>'
            
            test_results = results[test_type]
            for model_name, model_results in test_results.items():
                if not model_results:
                    continue
                    
                html += f'<div class="model-results"><h3>{model_name}</h3>'
                
                # Create summary table for this model
                html += '<table class="scores-table"><thead><tr><th>Prompt</th><th>Overall Score</th><th>Status</th></tr></thead><tbody>'
                
                for result in model_results:
                    prompt = result.get('prompt', 'N/A')[:50] + ('...' if len(result.get('prompt', '')) > 50 else '')
                    
                    if 'error' in result:
                        html += f'<tr><td>{prompt}</td><td class="error">Error</td><td class="error">{result["error"]}</td></tr>'
                    else:
                        overall_score = result.get('score', {}).get('Overall', 0)
                        score_class = self._get_score_class(overall_score)
                        html += f'<tr><td>{prompt}</td><td class="{score_class}">{overall_score:.2f}</td><td>✅ Success</td></tr>'
                        
                html += '</tbody></table></div>'
                
            html += '</div>'
            
        # Add summary section
        html += self._generate_summary_section(results)
        
        html += """
    <div class="summary">
        <h3>📝 Notes</h3>
        <ul>
            <li>Scores are on a scale of 0-5, where 5 is the best performance</li>
            <li>Green scores (4-5) indicate excellent performance</li>
            <li>Yellow scores (2-4) indicate moderate performance</li>
            <li>Red scores (0-2) indicate poor performance</li>
        </ul>
    </div>
</body>
</html>
"""
        
        return html
        
    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score value"""
        if score >= 4:
            return "score-good"
        elif score >= 2:
            return "score-medium"
        else:
            return "score-poor"
            
    def _generate_summary_section(self, results: Dict[str, Any]) -> str:
        """Generate summary section for HTML report"""
        html = '<div class="summary"><h2>📈 Summary</h2>'
        
        # Calculate average scores per model
        model_averages = {}
        
        for test_type in ['card_detection', 'person_reid']:
            if test_type not in results:
                continue
                
            test_results = results[test_type]
            for model_name, model_results in test_results.items():
                if not model_results:
                    continue
                    
                scores = []
                for result in model_results:
                    if 'score' in result and 'Overall' in result['score']:
                        scores.append(result['score']['Overall'])
                        
                if scores:
                    avg_score = sum(scores) / len(scores)
                    if model_name not in model_averages:
                        model_averages[model_name] = []
                    model_averages[model_name].append(avg_score)
                    
        # Create summary table
        if model_averages:
            html += '<table class="scores-table"><thead><tr><th>Model</th><th>Average Score</th><th>Performance</th></tr></thead><tbody>'
            
            for model_name, scores in model_averages.items():
                overall_avg = sum(scores) / len(scores)
                score_class = self._get_score_class(overall_avg)
                performance = "Excellent" if overall_avg >= 4 else "Good" if overall_avg >= 2 else "Needs Improvement"
                html += f'<tr><td><strong>{model_name}</strong></td><td class="{score_class}">{overall_avg:.2f}</td><td>{performance}</td></tr>'
                
            html += '</tbody></table>'
            
        html += '</div>'
        return html
        
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed comparison report between models"""
        # TODO: Implement detailed model comparison analysis 