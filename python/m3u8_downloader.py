import subprocess
import re
import binascii

url = '1.m3u8'
out = '1.mp4'
is_crypt = False
iv = ""
key = ""

def download_m3u8():
  if url.find("key") != -1:
    is_crypt = True
  else:
    is_crypt = False
    cmd_line = "wget -c %s -O tmp.m3u8" % (url)
    print(cmd_line)
    subprocess.call(cmd_line, shell=True)

#EXT-X-KEY:METHOD=AES-128,URI="/key.key",IV=0x2491928cf57b78b1ba6875e0052cf2cc
def get_key_and_iv():
  with open("tmp.m3u8", 'r') as f:
    lines = f.readlines()
    for line in lines:
      if line.find("EXT-X-KEY") != -1:
        pattern = re.compile(r'URI="(.*?)"')
        m = pattern.findall(line)
        key_url = m[0]
        cmd_line = "wget -c %s -O tmp.key" % (key_url)
        subprocess.call(cmd_line, shell=True)
        with open("tmp.key", 'rb') as f1:
          key_bin = f1.read()
          key = binascii.b2a_hex(key_bin).decode('utf-8')
          print(key)
        pattern = re.compile(r'IV=0x(.*?)')
        m = pattern.findall(line)
        iv = m[0]
        print(iv)

def download_ts():
  with open("tmp.m3u8", 'r') as f:
    lines = f.readlines()
    idx = 0
    for line in lines:
      if line.find(".ts") != -1:
        if is_crypt:
          cmd_line = "wget -c %s -O t_%d.ts" % (line, idx)
          subprocess.call(cmd_line, shell=True)
          cmd_line = "openssl aes-128-cbc -d -in t_{}.ts -out {}.ts \
              -nosalt -iv {} -K {}".format(idx, idx, iv, key)
          subprocess.call(cmd_line, shell=True)
        else:
          cmd_line = "wget -c %s -O %d.ts" % (line, idx)
          subprocess.call(cmd_line, shell=True)
        idx += 1
    return idx

def concat_ts_to_mp4(num):
  with open("filelists.txt", mode='w', newline='\n', encoding='UTF-8') as fhndl:
    for x in range(0, num):
      new_lines = "file '{}.ts'\n".format(x)
      fhndl.writelines(new_lines)
  cmd_line = "ffmpeg -f concat -i filelist.txt -c copy {}".format(out)
  subprocess.call(cmd_line, shell=True)

if __name__=='__main__':
  download_m3u8()
  if is_crypt:
    get_key_and_iv()
  ts_num = download_ts()
  concat_ts_to_mp4(ts_num)
