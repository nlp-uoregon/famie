'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/scripts/uninstall_app.py
'''
import os
import shutil


def main(config):
    upload_folder = config["DEFAULT"]["UPLOAD_FOLDER"]

    if os.path.exists(upload_folder):
        reply = input(f"Delete directory {upload_folder}? [y/[n]] ")
        if reply.lower().strip() == "y":
            shutil.rmtree(upload_folder)
    else:
        print("Doing nothing")


if __name__ == "__main__":
    main()
