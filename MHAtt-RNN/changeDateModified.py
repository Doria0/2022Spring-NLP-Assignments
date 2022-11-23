import os
import time


# 修改文件的【修改日期】
def set_file_time(filename, updatetime='now', access_time='now'):
    filename = os.path.abspath(filename)
    if updatetime == 'now':
        new_updatetime = time.time()
    else:
        new_updatetime = time.mktime(time.strptime(updatetime, "%Y-%m-%d %H:%M:%S"))
    if access_time == 'now':
        new_access_time = time.time()
    else:
        new_access_time = time.mktime(time.strptime(access_time, "%Y-%m-%d %H:%M:%S"))
    os.utime(filename, (new_access_time, new_updatetime))


# 修改一个文件夹内所有文件的【修改日期】
def set_all(path):
    for i in os.listdir(path):
        file = os.path.realpath(os.path.join(path, i))
        if os.path.isfile(file):
            set_file_time(file)
        elif os.path.isdir(file):
            set_all(file)


if __name__ == '__main__':
    path = r"D:\Anaconda3\pkgs\pefile-2021.9.3"  # 这里更换为你要修改的文件夹
    set_all(path)  # 修改以上文件夹内所有文件的日期