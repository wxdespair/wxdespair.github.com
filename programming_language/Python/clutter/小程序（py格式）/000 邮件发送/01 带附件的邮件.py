# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:32:44 2020

@author: Administrator
"""
import smtplib# ,uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

#发邮件相关参数
smtpsever='smtp.qq.com'
sender='2057047563@qq.com'        # 发送者
psw="sxyfxzmkrgbueijb"            # qq邮箱授权码，不同号码不同授权
receiver='2943999149@qq.com'      # 接收者
# port=465                          # 465为默认选取的SMTP端口，不用修改

filepath="E:\\demo.html"  #编辑邮件的内容
with open(filepath,'rb') as fp:    #读文件
    mail_body=fp.read()

#主题
msg=MIMEMultipart()
msg["from"]=sender
msg["to"]=receiver
msg["subject"]=u"这个我的主题"

#正文
body=MIMEText(mail_body,"html","utf-8")
msg.attach(body)
att = MIMEText(mail_body,"base64","utf-8")
att["Content-Type"] = "application/octet-stream"
att["Content-Disposition"] = 'attachment; filename="test_report.html"'
msg.attach(att)

try:
    smtp=smtplib.SMTP()
    smtp.connect(smtpsever)                     #连接服务器
    smtp.login(sender,psw)
except:
    smtp=smtplib.SMTP_SSL(smtpsever)# ,port)
    smtp.login(sender,psw)  #登录
smtp.sendmail(sender,receiver,msg.as_string())  #发送
smtp.quit()