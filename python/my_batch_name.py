
new_file='filelist.txt'

with open(new_file, mode='w', newline='\n', encoding='UTF-8') as fhndl:
    for x in range(0, 203):
        new_lines = "file '{}.ts'\n".format(x)
        # new_lines = "{}.ts\n".format(x)
        fhndl.writelines(new_lines)

# cbbfd7e3517f764b56df53f1f293e54f
