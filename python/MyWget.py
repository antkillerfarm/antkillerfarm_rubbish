import os
import subprocess
import re

def mywget():
  exe_path = ""
  exe_name = "wget"
  urls_file = '1.m3u8'
  pattern = re.compile(r'img src="(.*?)"')

  id = 0
  with open(urls_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
      # print(line)
      if line.find("http") != -1 and line.find(".ts") != -1:
        #line = line.replace('http','https')
        cmd_line = "%s -c %s -O %d.ts" % (exe_path + exe_name,
                                          line[0:-1], id)
        id = id + 1
        print(cmd_line)
        subprocess.call(cmd_line, shell=True)


def myssl():
  exe_path = ""
  exe_name = "openssl"
  urls_file = 'filelist.txt'
  pattern = re.compile(r'img src="(.*?)"')
  iv = "2491928cf57b78b1ba6875e0052cf2cc"
  key = "cbbfd7e3517f764b56df53f1f293e54f"

  with open(urls_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
      # print(line)
      cmd_line = "{} aes-128-cbc -d -in {} -out t_{} -nosalt -iv {} -K {}".format(
                  exe_path + exe_name, line[0:-1], line[0:-1], iv, key)
      print(cmd_line)
      subprocess.call(cmd_line, shell=True)

if __name__=='__main__':
  # mywget()
  myssl()
