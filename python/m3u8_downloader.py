import subprocess
import re
import binascii
from concurrent.futures import ProcessPoolExecutor

url = '"index.m3u8"'
out = 'a44.mp4'
is_crypt = False
iv = ""
key = ""

def download_m3u8():
  global is_crypt
  if url.find("key") != -1:
    is_crypt = True
  else:
    is_crypt = False
  cmd_line = "wget -c %s -O tmp.m3u8" % (url)
  print(cmd_line)
  subprocess.call(cmd_line, shell=True)

#EXT-X-KEY:METHOD=AES-128,URI="/key.key",IV=0x2491928cf57b78b1ba6875e0052cf2cc
def get_key_and_iv():
  global key, iv
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
          print("key=" + key)
        pattern = re.compile(r'IV=0x(.*)')
        m = pattern.findall(line)
        iv = m[0]
        print("iv=" + iv)

def get_base_path():
  pos = url.rfind("/")
  return url[1:pos + 1]

def download_handler(url0, idx):
  if is_crypt:
    cmd_line = 'wget -c "%s" -O t_%d.ts' % (url0, idx)
    subprocess.call(cmd_line, shell=True)
    cmd_line = "openssl aes-128-cbc -d -in t_{}.ts -out {}.ts -nosalt -iv {} -K {}".format(idx, idx, iv, key)
    print(cmd_line)
    subprocess.call(cmd_line, shell=True)
  else:
    base_path = get_base_path()
    cmd_line = 'wget -c "%s" -O %d.ts' % (base_path + url0, idx)
    print(cmd_line)
    subprocess.call(cmd_line, shell=True)

def download_ts():
  with open("tmp.m3u8", 'r') as f:
    lines = f.readlines()
    idx = 0
    download_task_list = {}

    for line in lines:
      if line.find(".ts") != -1:
        print(line)
        download_task_list[line[:-1]] = idx
        idx += 1

    print(len(download_task_list))
    p=ProcessPoolExecutor(4)
    for key, value in download_task_list.items():
      p.submit(download_handler, key, value)
    p.shutdown(wait=True)
    return idx

def concat_ts_to_mp4(num):
  with open("filelists.txt", mode='w', newline='\n', encoding='UTF-8') as fhndl:
    for x in range(0, num):
      new_lines = "file '{}.ts'\n".format(x)
      fhndl.writelines(new_lines)
  cmd_line = "ffmpeg -f concat -i filelists.txt -c copy {}".format(out)
  subprocess.call(cmd_line, shell=True)

if __name__=='__main__':
  download_m3u8()
  if is_crypt:
    get_key_and_iv()
  ts_num = download_ts()
  concat_ts_to_mp4(ts_num)
