# run_report_generator.py
"""
Usage script for CortexX Report Generator
Run this script to generate the professional technical analysis report
"""

import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def main():
    """Main execution function"""
    print("="*60)
    print("CORTEXX REPORT GENERATOR - SETUP & EXECUTION")
    print("="*60)
    
    # Install requirements if needed
    response = input("\nDo you want to install required packages? (y/n): ")
    if response.lower() == 'y':
        if not install_requirements():
            print("Failed to install requirements. Please install manually:")
            print("pip install python-docx pandas lxml")
            return 1
    
    # Import and run the generator
    try:
        from cortexx_report_generator import CortexXReportGenerator, main as generate_main
        
        print("\nStarting report generation...")
        print("-"*40)
        
        # Generate the report
        result = generate_main()
        
        if result == 0:
            print("\n" + "="*60)
            print("SUCCESS: Report generated successfully!")
            print("="*60)
            print("\nThe report includes:")
            print("• 10 comprehensive sections with professional formatting")
            print("• Executive summary with grading tables")
            print("• Business impact analysis with ROI calculations")
            print("• Technology stack recommendations")
            print("• Implementation roadmap")
            print("\nThe document is ready for executive presentation.")
        else:
            print("\nFailed to generate report.")
            return 1
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all requirements are installed.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())