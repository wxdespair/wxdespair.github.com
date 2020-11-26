# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:55:31 2020

@author: Administrator
"""
import requests
from bs4 import BeautifulSoup

headers = { # 伪装请求头
    'cookie':'__cfduid=d71dbda9330454ae39e77b908740f3d041582867701; security_session_verify=eaf3d1c1dec9c2062217aed8ca51692a',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36'
}

count = 10
num = 60568852
i = 1

while (count>0):
    # 处理url并获取文本
    url = 'https://www.bshuwu.com/nov/367/367784/'+str(num)+'.html'
    print('\t'+url+'\n')
    html = requests.get(url,headers=headers)
    html.encoding='gbk' # 设置文本解码方式
    data = html.text
    print('\t文本爬取完毕！\n')
    # 正文文本在标签<div id="content"></div>内，章节标题在标签<h1></h1>内
    soup = BeautifulSoup(data,"lxml")
    chapter = soup.select('h1')  #h1可以改成任意标签
    test_text = soup.select('#content')
                            # test_text = soup.find(id="content")
                            
    # 文本储存                        
    f = open("text.txt","a+",encoding="utf-8")
    f.writelines(chapter[0].text[5:]+"\n") # 去除章节名前部的  章节目录 
    f.writelines(test_text[0].text+"\n") 
    f.close()
    print('\t文本存储完毕！\n')
    i = i+1
    print('\t第' + str(i) + '章下载成功！\n')
    print(str(count)+'\n')
    num += 1
    count -= 1
print('完毕！')
num = 60568852
i = 1
    
    
    
    
    
    
    
'''
url = 'https://www.bshuwu.com/nov/367/367784/60568853.html'
html = requests.get(url)#,headers=headers)
html.encoding='gbk' 
    # 设置文本解码方式
data = html.text

# 主要文本在标签<div id="content"></div>内，章节标题在标签<h1></h1>内
soup = BeautifulSoup(data,"lxml")
chapter = soup.select('h1')  #h1可以改成任意标签
test_text = soup.select('#content')
# test_text = soup.find(id="content")                   

f = open("text.txt","a+",encoding="utf-8")
f.writelines(chapter[0].text[5:]+"\n") 
f.writelines(test_text[0].text+"\n") 
f.close()                         
'''








'''
headers = { # 伪装请求头
        'Referer':'https://www.bshuwu.com/nov/367/367784/60568852.html',
        'Sec-Fetch-User':'?1',
        'Upgrade-Insecure-Requests':'1',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
    }
'''



'''
import requests
url = 'https://www.bshuwu.com/nov/367/367784/60568853.html'

headers = { # 伪装请求头
        'Referer':'https://www.bshuwu.com/nov/367/367784/60568852.html',
        'Sec-Fetch-User':'?1',
        'Upgrade-Insecure-Requests':'1',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
    }

proxy = '101.132.118.170:8080'
proxies = {
    "http": "http://101.132.118.170:8080/",
    "https": "http://101.132.118.170:8080/"
}
data = {"info": "biu~~~ send post request"}

# html = requests.get(url,headers=headers)
html = requests.post(url,headers=headers,data=data,proxies=proxies)
html.encoding='gbk' 
    # 设置文本解码方式
data = html.text
print(html.status_code) 
    # 返回状态码
print(data)

'''