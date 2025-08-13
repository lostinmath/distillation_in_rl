import os
import subprocess


def write_git_id_to_file(file_path: str):
    """
    Writes the current Git commit ID to a specified file.

    Args:
        file_path (str): The path to the file where the Git commit ID will be written.
    """
    try:
        

        current_dir = os.path.basename(os.getcwd())
        print(f"Current dir {current_dir}")
        if current_dir == "piscenco":
            git_diff = (
                subprocess.check_output(
                    ["sh", "-c", "cd thesis_octo/thesis-code/ && git rev-parse HEAD&& cd ../"]
                )
                .strip()
                .decode("utf-8")
            )
        elif current_dir == "thesis-code":
            git_diff = (
                subprocess.check_output(["sh", "-c", "git rev-parse HEAD"])
                .strip()
                .decode("utf-8")
            )
        elif current_dir == "thesis_octo":
            git_diff = (
                subprocess.check_output(
                    ["sh", "-c", "cd thesis-code/ && git rev-parse HEAD&& cd ../"]
                )
                .strip()
                .decode("utf-8")
            )
        else:
            print("Not in the right directory. GitInfo was not written.")
            return

        # Write the Git diff to the specified file
        with open(file_path, "w") as file:
            file.write(git_diff)

        print(f"Git id written to {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error obtaining id diff: {e}")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")


def write_git_diff_to_file(file_path: str):
    """
    Writes the current Git diff to a specified file.

    Args:
        file_path (str): The path to the file where the Git diff will be written.
    """
    try:

        current_dir = os.path.basename(os.getcwd())
        print(f"Current dir {current_dir}")
        if current_dir == "piscenco":
            git_diff = (
                subprocess.check_output(
                    ["sh", "-c", "cd thesis_octo/thesis-code/ && git diff && cd ../"]
                )
                .strip()
                .decode("utf-8")
            )
        elif current_dir == "thesis-code":
            git_diff = (
                subprocess.check_output(["sh", "-c", "git diff"])
                .strip()
                .decode("utf-8")
            )
        elif current_dir == "thesis_octo":
            # Run the git diff command
            git_diff = (
                subprocess.check_output(
                    ["sh", "-c", "cd thesis-code/ && git diff && cd ../"]
                )
                .strip()
                .decode("utf-8")
            )
        else:
            print("Not in the right directory. GitDiff was not written.")
            return

        # Write the Git diff to the specified file
        with open(file_path, "w") as file:
            file.write(git_diff)

        print(f"Git diff written to {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error obtaining Git diff: {e}")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")