import os
import sys
import datetime
date_ = datetime.datetime.today().isoformat()[0:10]
date = date_[:4] + '.' + date_[5:7] + '.' + date_[8:]

if __name__ == '__main__':
    file_name = sys.argv[0]  # 执行文件的名字
    if len(sys.argv) > 1:
    	commit = sys.argv[1]  # 提交附带信息
    else:
    	commit = date
    print('********** start add file **********')
    os.system('git add .')
    print('*********** start commit ***********')
    os.system('git commit -m\'%s\'' % commit)
    print('commit : %s' % commit)
    print('************ start push ************')
    os.system('git push')
