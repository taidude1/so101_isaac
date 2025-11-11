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

    # run other commands for setup
    run_start_commands = json.loads(args.run_start_commands)
    for cmd in run_start_commands:
        subprocess.run(cmd, shell=True)

    # install dependencies of new modules
    for mount_point in file_mounts.values():
        if os.path.exists(os.path.join(mount_point, "setup.py")):
            subprocess.run([sys.executable, "-m", "pip", "install", "--editable", mount_point])

    # run train script
    job_script = args.job_script
    print(f"Executing {sys.executable} on {job_script} with args {remaining_args}")
    subprocess.run([sys.executable, job_script, *remaining_args])

    # clean up copied directories
    for mount_point in file_mounts.values():
        if os.path.isdir(mount_point):
            shutil.rmtree(mount_point)
        else:
            os.remove(mount_point)
