import os
import subprocess

def myconvert():
  exe_path = ""
  exe_name = "convert"
  img_path = "/home/tj/my/temp/1/"
  des_path = "/home/tj/my/temp/2/"
  for img_file in os.listdir(img_path):
    cmd_line = "%s %s -resize 50%% %s" % (exe_path + exe_name,
                                        img_path + img_file,
                                        des_path + img_file)
    print(cmd_line)
    subprocess.call(cmd_line, shell=True)

if __name__=='__main__':
  myconvert()
