import os
import subprocess
import re

def mywget():
  exe_path = ""
  exe_name = "wget"
  urls_file = 'url_2.txt'
  pattern = re.compile(r'img src="(.*?)"')

  with open(urls_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
      urls = pattern.findall(line)
      for url in urls:
        url = url.replace('http','https')
        cmd_line = "%s %s" % (exe_path + exe_name,
                                        url)
        print(cmd_line)
        subprocess.call(cmd_line, shell=True)


if __name__=='__main__':
  mywget()
