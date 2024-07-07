import subprocess

def run_script(script_path):
    """Helper function to run a script using subprocess."""
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}: {result.stderr}")
    else:
        print(f"Successfully ran {script_path}: {result.stdout}")

def main():
    # Run final_version_smoother.py
    print("Running final_version_smoother.py...")
    run_script("animation/final_trail_smoother.py")
    
    # Run simulation_minibatch.py
    print("Running simulation_minibatch.py...")
    run_script("simulation_minibatch.py")
    
    # Run measuring_nll.py
    print("Running measuring_nll.py...")
    run_script("measuring_nll.py")
    
    # Run generate_predictions_PBNN.py
    print("Running generate_predictions_PBNN.py...")
    run_script("generate_predictions_PBNN.py")
    
    # Run plot_predictions.py
    print("Running plot_predictions.py...")
    run_script("plot_predictions.py")

if __name__ == "__main__":
    main()
