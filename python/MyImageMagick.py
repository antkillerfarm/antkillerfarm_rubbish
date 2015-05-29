import os
import subprocess

def myconvert():
  exe_path = "F:\\setup\\ImageMagick-6.9.1-3\\"
  exe_name = "convert.exe"
  img_path = "H:\\my\\work\\sample\\antkillerfarm_rubbish\\python\\img\\1\\"
  des_path = "H:\\my\\work\\sample\\antkillerfarm_rubbish\\python\\img\\2\\"
  for img_file in os.listdir(img_path):
    cmd_line = "%s %s -resize 50%% %s" % ("\"" + exe_path + exe_name + "\"",
                                        "\"" + img_path + img_file + "\"",
                                        "\"" + des_path + img_file + "\"")
    print(cmd_line)
    subprocess.Popen(cmd_line)

if __name__=='__main__':
  myconvert()
