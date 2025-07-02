#!/usr/bin/env python3
"""
Environment Setup Script
Helps users set up their environment for the AI Vision Benchmark
"""

import os
import sys
from pathlib import Path
import shutil
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'Pillow', 'opencv-python', 
        'matplotlib', 'yaml', 'openai'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    if missing_packages:
        logger.info(f"To install missing packages: pip install {' '.join(missing_packages)}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/images',
        'data/videos', 
        'results',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def create_config_file():
    """Create config file from template if it doesn't exist"""
    config_path = Path('config/config.yaml')
    template_path = Path('config/config.template.yaml')
    
    if not config_path.exists() and template_path.exists():
        shutil.copy(template_path, config_path)
        logger.info("Created config.yaml from template")
        logger.warning("Please edit config.yaml to add your API keys")
    elif config_path.exists():
        logger.info("Config file already exists")
    else:
        logger.error("Config template not found")

def check_git():
    """Check if git is available"""
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
        logger.info("✓ Git is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("✗ Git not found - version control won't be available")
        return False

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Config files with API keys
config/config.yaml

# Data files
data/

# Results
results/

# Logs
logs/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content.strip())
        logger.info("Created .gitignore file")

def main():
    """Main setup function"""
    logger.info("🚀 Setting up AI Vision Benchmark environment...")
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Create config file
    create_config_file()
    
    # Check git and create .gitignore
    if check_git():
        create_gitignore()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    logger.info("\n" + "="*50)
    logger.info("Setup completed!")
    
    if not deps_ok:
        logger.warning("Please install missing dependencies before running benchmarks")
        logger.info("Run: pip install -r requirements.txt")
    
    logger.info("Next steps:")
    logger.info("1. Edit config/config.yaml with your API keys")
    logger.info("2. Add test images to data/images/")
    logger.info("3. Add test videos to data/videos/")
    logger.info("4. Run: python run_benchmark.py")
    logger.info("="*50)

if __name__ == "__main__":
    main() 