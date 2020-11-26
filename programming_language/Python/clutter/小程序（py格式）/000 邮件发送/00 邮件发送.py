# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import smtplib# ,uuid
from email.mime.text import MIMEText

#发邮件相关参数
smtpsever='smtp.qq.com'
sender='2057047563@qq.com'        # 发送者
psw="sxyfxzmkrgbueijb"            # qq邮箱授权码，不同号码不同授权
receiver='2943999149@qq.com'      # 接收者
# port=465                          # 465为默认选取的SMTP端口，不用修改


#编辑邮件内容
subject = u"你猜这是啥？"          # 邮件主题
body = str("哈喽")                # 邮件正文
msg=MIMEText(body,'html','utf-8')
msg['from']=sender
msg['to']=receiver
msg['subject'] = subject

#链接服务器发送
smtp = smtplib.SMTP_SSL(smtpsever) #,port)
smtp.login(sender,psw)                          #登录
smtp.sendmail(sender,receiver,msg.as_string())  #发送
smtp.quit()                                     #关闭