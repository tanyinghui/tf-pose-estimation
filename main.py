import cv2
import time
import mysql.connector

from datetime import datetime

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

URLS = ['rtsp://admin:Admin123@192.168.1.100/Streaming/channels/101',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/201',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/301',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/401',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/501',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/601',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/701',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/801',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/901',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/1001',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/1101',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/1201',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/1301',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/1401',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/1501',
        'rtsp://admin:Admin123@192.168.1.100/Streaming/channels/1601']
HOST='localhost'
USER='lauretta'
PASSWORD='password'
DATABASE='SIT'

mydb = mysql.connector.connect(
    host=HOST,
    user=USER,
    passwd=PASSWORD,
    database=DATABASE
)

mycursor = mydb.cursor()

def generate_dbcol_name(url):
    idx = url[-4:-2]
    if idx[0] == '/':
        idx = idx.replace('/', '0')
    return ('cam'+str(idx))

def update_database_activitylog(url, log, db, cursor):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cam = generate_dbcol_name(url)
    sql = "INSERT INTO `SIT`.`activitylog`(`" + str(cam) + "`,`timestamp`) VALUES ('" + str(log) + "','" + timestamp + "');"
    cursor.execute(sql)
    db.commit()

def update_database_summarylog(url, cnt, db, cursor):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cam = generate_dbcol_name(url)
    sql = "INSERT INTO `SIT`.`summarylog`(`mainzone`,`timestamp`,`count`) VALUES ('" + cam + "','" + timestamp + "','"+ str(cnt) +"');"
    cursor.execute(sql)
    db.commit()
    print(str(timestamp) + ' AT ' + cam)

w, h = model_wh('0x0')
e = TfPoseEstimator(get_graph_path("mobilenet_v2_large"), target_size=(432, 368))

while True:
    for idx, url in enumerate(URLS):
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
        time.sleep(5)
        log = []
        if cap.isOpened():
            ret, image = cap.read()
            if image is None:
                print(url + "is blank")
                continue
            image_h, image_w = image.shape[:2]
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
            
            for human in humans:
                centers = {}
                for i in range(common.CocoPart.Background.value):
                    if i not in human.body_parts.keys():
                        continue
                    body_part = human.body_parts[i]
                    center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                    centers[i] = center
                log.append(centers)
                
        if len(humans) > 1:
            update_database_activitylog(url, log, mydb, mycursor)
            update_database_summarylog(url, len(humans), mydb, mycursor)
