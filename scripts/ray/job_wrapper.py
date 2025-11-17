import argparse
import sys
import shutil
import subprocess
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--job-script", type=str, help="Job script to run.")
parser.add_argument(
    "--file-mounts",
    type=str,
    default=None,
    help=("Dictionary of python modules to mount to specific directories."),
)
parser.add_argument(
    "--run-start-commands",
    type=str,
    default=None,
    help=("List of commands to execute at the start of the run."),
)
args, remaining_args = parser.parse_known_args()


if __name__ == "__main__":
    # copy all python_packages into the isaaclab source/ dir
    file_mounts = json.loads(args.file_mounts)
    python_packages = [p for p in sys.path if "/py_modules_files/" in p]
    assert len(python_packages) == len(file_mounts), (
        "Mismatched length between python_packages and file_mounts -- configuration may be corrupted!"
    )
    for p in python_packages:
        name = os.listdir(p)[0]
        shutil.copytree(os.path.join(p, name), file_mounts[name], dirs_exist_ok=True)
    print(f"Created {len(file_mounts)} file mounts.")

    # run other commands for setup
    print(f"Running start commands: {args.run_start_commands}")
    run_start_commands = json.loads(args.run_start_commands)
    for cmd in run_start_commands:
        subprocess.run(cmd, shell=True)

    # install dependencies of new modules
    print("Installing external dependencies...")
    total_installed = 0
    for mount_point in file_mounts.values():
        if os.path.exists(os.path.join(mount_point, "setup.py")):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--editable",
                    mount_point,
                    "--quiet",
                    "--disable-pip-version-check",
                    "--root-user-action=ignore",
                ]
            )
            total_installed += 1
    print(f"Installed dependencies at {total_installed} mount points.")

    # run train script
    job_script = args.job_script
    print(f"Executing {sys.executable} on {job_script} with args {remaining_args}")
    try:
        subprocess.run([sys.executable, job_script, *remaining_args], check=True)
    finally:
        # clean up copied directories
        for mount_point in file_mounts.values():
            if os.path.isdir(mount_point):
                shutil.rmtree(mount_point)
            else:
                os.remove(mount_point)
        print(f"Cleaned up {len(file_mounts)} file mounts.")
