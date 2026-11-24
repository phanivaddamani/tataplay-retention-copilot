"""run_all.py -- End-to-end execution script for the entire Tata Play Copilot capstone."""
import os
import sys
import subprocess

def run_step(cmd_desc, cmd_args):
    print(f"\n=======================================================")
    print(f"RUNNING: {cmd_desc}")
    print(f"Command: {' '.join(cmd_args)}")
    print(f"=======================================================")
    result = subprocess.run([sys.executable] + cmd_args, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed with exit code {result.returncode}")

def main():
    print("Starting End-to-End Tata Play Copilot Pipeline...")
    run_step("1. Data Generation (generate_data.py)", ["generate_data.py"])
    run_step("2. Part A: Audit, Clean, Store, Query (src/part_a_data.py)", ["src/part_a_data.py"])
    run_step("3. Part B: Churn Modeling & Threshold Tuning (src/part_b_churn.py)", ["src/part_b_churn.py"])
    run_step("4. Part C: Segments, Spend Model & Revenue at Risk (src/part_c_segments.py)", ["src/part_c_segments.py"])
    run_step("5. Part D Task 1: LLM Client Demonstrations (src/test_llm_client.py)", ["src/test_llm_client.py"])
    run_step("6. Part D Tasks 2-5: Prompts, Extraction, RAG, Copilot (src/part_d_copilot.py)", ["src/part_d_copilot.py"])
    print("\n=======================================================")
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == "__main__":
    main()
