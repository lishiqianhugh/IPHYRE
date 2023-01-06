import os
import sys
import smtplib
import time
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

from iphyre.utils import collect_play_all

# config
try:
    player_name = sys.argv[1]
except IndexError:
    raise ValueError('Please specify your name in the command! such as: \npython collect_play_all.py your_name')

folder_path = './player_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

save_path = f'{folder_path}/{player_name}.json'

# play
t1 = time.time()
collect_play_all(player_name, max_episode=5, save_path=save_path)
t2 = time.time()

# send results to our mail
host_server = 'smtp.qq.com'
sender_qq = '2816425039@qq.com'
pwd = 'hbbtirqcnpyidgfc'
receiver = '2816425039@qq.com'
mail_title = f'iphyre_{player_name}'

mail_content = f'Play Duration: {t2 - t1}'
msg = MIMEMultipart()
msg["Subject"] = Header(mail_title,'utf-8')
msg["From"] = sender_qq
msg["To"] = Header("data mail","utf-8")

msg.attach(MIMEText(mail_content,'html'))
filename = f'{player_name}.json'
atta = MIMEText(open(save_path, 'rb').read(), 'base64', 'utf-8')
atta["Content-Disposition"] = f'attachment; filename="{filename}"'

msg.attach(atta)


try:
    smtp = SMTP_SSL(host_server)
    smtp.set_debuglevel(0)
    smtp.ehlo(host_server)
    smtp.login(sender_qq,pwd)
    smtp.sendmail(sender_qq,receiver,msg.as_string())
    smtp.quit()
    print("Submit successfully!")
except smtplib.SMTPException:
    print("Submission fail.")
