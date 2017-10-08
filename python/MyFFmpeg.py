import os
import subprocess

def myconvert():
  exe_path = ""
  exe_name = "ffmpeg"
  src_path = "/home/data/my/video/1/"
  des_path = "/home/data/my/video/2/"
  for src_file in os.listdir(src_path):
    cmd_line = "%s -i %s -s 960x540 -acodec copy %s" % (exe_path + exe_name,
                                        src_path + src_file,
                                        des_path + src_file)
    print(cmd_line)
    subprocess.call(cmd_line, shell=True)

if __name__=='__main__':
  myconvert()
