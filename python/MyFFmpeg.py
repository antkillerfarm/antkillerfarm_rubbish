import os
import subprocess

def myconvert():
  exe_path = ""
  exe_name = "ffmpeg"
  src_path = "/media/tangjing/新加卷/information/myinf/pic/2021/video/"
  des_path = "/media/tangjing/新加卷/information/myinf/pic/2021/video2/"
  for src_file in os.listdir(src_path):
    cmd_line = "%s -i %s -vf scale=iw*.5:ih*.5 -acodec copy %s" % (exe_path + exe_name,
                                        src_path + src_file,
                                        des_path + src_file)
    print(cmd_line)
    subprocess.call(cmd_line, shell=True)

if __name__=='__main__':
  myconvert()
