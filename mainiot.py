#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:40:49 2019

@author: davidfauzi
"""
import multiprocessing
from Adafruit_IO import Client, Feed, RequestError
import time
import requests
import base64
import os
import serial
import string
import pynmea2
import json
import numpy as np
import PIL
try:
    import Image
except ImportError:
    from PIL import Image

#Set up GPS

#Set up GPS


# Set to your Adafruit IO key.
# Remember, your key is a secret,
# so make sure not to publish it when you publish this code!
ADAFRUIT_IO_KEY = '0f7873c4c7bf48bb95e79d4112889e8e'

# Set to your Adafruit IO username.
# (go to https://accounts.adafruit.com to find your username)
ADAFRUIT_IO_USERNAME = 'davidfauzi'

# Create an instance of the REST client
aio = Client(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)
print("connectinng to MQTT server...")
#DEFINE ALL THE FEEDS
# 
# temp=aio.feeds('temp')
# wind=aio.feeds('wind')
# cam = aio.feeds('cam')
# basex=aio.feeds('basex')
# basey=aio.feeds('basey')
# crowdcar=aio.feeds('crowdcar')
# crowdppl=aio.feeds('crowdppl')
# humidity=aio.feeds('humidity')
feed_cam=aio.feeds('cam')
feed_carcrowd=aio.feeds('carcrowd')
feed_peoplecrowd=aio.feeds('pplcrowd')
feed_temp=aio.feeds('temp')
feed_hum=aio.feeds('hum')
feed_cam=aio.feeds('cam')
feed_count=aio.feeds('count')
""" requests part"""
print("setting up feeds done")
def getgps():
    loctxt=open("location.txt","r")
    if(loctxt.mode=='r'):
        latdata,londata,temp,hum=(loctxt.read()).split(',')
    loctxt.close()
    if(True):
        latdata=-6.886030
#         latdata=-7.765995
    if(True):
        londata=107.619385
#         londata=110.371666
    return latdata,londata,hum,temp
counts=0
#doesnt used anymore 
# def getlonlat():
#     #request and get IP Data info 
#     res = requests.get('https://ipinfo.io/')
#     data = res.json()
#     
#     location = data['loc'].split(',')
#     latitude = location[0]
#     longitude = location[1]
#     return latitude,longitude

def getweather():
    latitude,longitude= getgps()  
    lastlat,lastlon=latitude,longitude 
    #print("lat :" + str(latitude) + "lon :" + str(longitude)) 
    url = 'http://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid=2acb4f4cd11793ac872371fa053e4a13&units=metric'.format(latitude, longitude)
    res = requests.get(url)
    data = res.json()
    #print(data)
    temp = data['main']['temp']
    wind_speed = data['wind']['speed']
    humidity = data['main']['humidity']
    latitude = data['coord']['lat']
    longitude = data['coord']['lon']
    print("sucessfully get loc and weather data")
    return temp,humidity
#    gsheet(temp,hum,xpos,xmin,ypos,ymin,cden,pden)
def gsheet(temp,hum,xpos,xmin,ypos,ymin,cden,pden,count):
    in1 =str(xmin)+"|||"+str(xpos)+ "|||" + str(ymin)
    in2 =str(ypos)+"|||"+str(pden)+"|||"+str(cden)
    in3 =str(count)+"|||"+str(temp)+"|||"+str(hum)
    print("IFTTT Success")
    r = requests.post('https://maker.ifttt.com/trigger/tes/with/key/elRjQawsd4tGh-upbaMZeMmCSPySp4dXW6Y3OHSqJ9i', params={"value1":in1,"value2":in2,"value3":in3})
    
def sendtemp(temp):
    try:
        aio.send('temp',temp)
        print("temp published :"+str(temp))
    except:
        print("temp failed")
    time.sleep(0.1)
def sendhum(hum,lat,lon):
    ele=3
    lat=40.7261
    lon=-74.005334
    try:
        metadata = { 'lat':lat, 'lon':lon, 'ele':ele, 'created_at':time.asctime(time.gmtime()) }
        print(metadata)
#         aio.send_data(feed_hum.key,hum,metadata)
        aio.send('hum',hum)
        print("success published hum :"+str(hum))
    except:
        print("hum and loc failed")
    time.sleep(0.1)

def sendpden(pden):
    try:
        #metadata = { 'lat':4, 'lon':3, 'created_at':time.asctime(time.gmtime()) }
        #aio.send_data(feed_peoplecrowd.key,float(pden),metadata)
        aio.send('pplcrowd',pden)
        print("crowdppl published : "+str(pden))
        #print("lon lat published : "+str(lon)+ " " + str(lat))
        
    except:
        print("crowdppl failed")
    time.sleep(0.1)
def sendcount(count):
    try:
        #metadata = { 'lat':4, 'lon':3, 'created_at':time.asctime(time.gmtime()) }
        #aio.send_data(feed_peoplecrowd.key,float(pden),metadata)
        aio.send('count',count)
        print("count published : "+str(count))
        #print("lon lat published : "+str(lon)+ " " + str(lat))
        
    except:
        print("count failed")
    time.sleep(0.1)
    
def sendcden(cden):
    try:
        aio.send('carcrowd',cden)
        print("crowdcar published: "+str(cden))
    except:
        print("crowdcar failed")
    time.sleep(0.1)
    
def senddata(temp,hum,lat,lon):
    ele=6
    #SEND TEMP
    try:
        aio.send('temp',temp)
        print("temp published :"+str(temp))
    except:
        print("temp failed")
    time.sleep(0.1)
    #SEND HUM + Location 
    try:
        metadata = { 'lat':float(lat), 'lon':float(lon), 'ele':float(ele), 'created_at':time.asctime(time.localtime()) }
#         aio.send_data(humidity.key,hum,metadata)
        aio.send('hum',hum)
        print("hum :"+str(hum)+" loc : "+str(lat)+","+str(lon))
    except:
        print("hum and loc failed")
    time.sleep(0.1)
    #SEND WIND
    try:
        aio.send('wind',wind)
        print("wind published :"+ str(wind))
    except:
        print("wind failed")
    time.sleep(0.1)
    
    
def sendcam():
    #SEND IMAGE

    list_im = ['send.jpg', 'vector.jpg']
    imgs    = [ PIL.Image.open(i) for i in list_im ]
    #pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

    #save that beautiful picture
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save( 'Trifecta.jpg' )    

#     for a vertical stacking it is simple: use vstack
#     imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
#     imgs_comb = PIL.Image.fromarray( imgs_comb)
#     imgs_comb.save( 'Trifecta_vertical.jpg' )
    """
    Plan: Asumsi sudah ada gambar send.jpg, dari situ disini diubah base64 lalu dikirim ke cam
    """
    with open("Trifecta.jpg", "rb") as imageFile:
        image = base64.b64encode(imageFile.read())
        # encode the b64 bytearray as a string for adafruit-io
        image_string = image.decode("utf-8")
    try:
      aio.send('cam', (image_string))
      print('Picture sent to Adafruit IO')
      print(len(image_string))
    except:
      print('Sending to Adafruit IO Failed...')
    time.sleep(0.1)


def opentext():
    with open('jsonfile.json') as f:
        data=json.load(f)
    xpos=data["xplus"]
    xmin=data["xmin"]
    ymin=data["ymin"]
    ypos=data["yplus"]
    crowdcarcount=data["crowdcar"]
    crowdpplcount=data["crowdppl"]
    count=data["count"]
    print("JSON loaded")
    return xpos,xmin,ymin,ypos,crowdpplcount,crowdcarcount,count
# def sendopencv():
#     #SEND OPENCV DATA HERE
#     basisx,basisy,pplcrowd,carcrowd=opentext()
#     try:
#         aio.send('basex',basisx)
#         print("basex published : "+str(basisy))
#     except:
#         print("basex failed")
#     time.sleep(0.1)
#     try:
#         aio.send('basey',basisy)
#         print("basey published : "+str(basisy))
#     except:
#         print("basey failed")
#     time.sleep(0.1)
#     try:
#         aio.send('crowdppl',pplcrowd)
#         print("crowdppl published : "+str(pplcrowd))
#     except:
#         print("crowdppl failed")
#     time.sleep(0.1)
#     try:
#         aio.send('crowdcar',carcrowd)
#         print("crowdcar published: "+str(carcrowd))
#     except:
#         print("crowdcar failed")
#     time.sleep(0.1)
#     print("Send Opencv done")
#     return basisx,basisy,pplcrowd,carcrowd
def maincode():
    #ambil data
#     lat,lon=getgps()
#     xpos,xmin,ypos,ymin,pden,cden,count=opentext()
#     temp,hum= getweather()
#     #kirim data
#     gsheet(temp,hum,xpos,xmin,ypos,ymin,cden,pden,count)
#     sendtemp(temp)
#     sendhum(hum,lat,lon)
#     sendpden(pden)
#     sendcden(cden)
#     sendcount(count)
    x=time.monotonic()
    counts=1
    while((counts%8)!=0):
        sendcam()
        counts=counts+1
        time.sleep(2.2)
        print("counts now:"+str(counts))
    print('')
#     print("SENDING DATA")
#     lat,lon,hum,temp=getgps()
#     xpos,xmin,ypos,ymin,pden,cden,count=opentext()
#     print(hum)
#     print(temp)
#     print(lat)
#     print(lon)
#     print(count)
#     gsheet(temp,hum,xpos,xmin,ypos,ymin,cden,pden,count)
#     sendtemp(temp)
#     time.sleep(0.5)
#     sendhum(hum,lat,lon)
#     time.sleep(0.5)
#     sendpden(pden)
#     time.sleep(0.5)
#     sendcden(cden)
#     time.sleep(0.5)
#     sendcount(count)
    print("time elapsed")
    print(time.monotonic()-x)

"""
    tx=time.monotonic()
    sendpic()
    temp,wind,hum,lat,lon=getweather()
    senddata(temp,wind,hum,lat,lon)
    baseX,baseY,ppl,car=sendopencv()
    iftttpush(temp,wind,hum,baseX,baseX,car,ppl)
    print("1 iteration sucess, looping")
    print("duration  :",time.monotonic()-tx)
"""

#MAIN LOOP
while(True):
    if __name__ == '__main__':
        p=multiprocessing.Process(target=maincode)
        p.start()
        p.join(30)
        if p.is_alive():
            print("takes too long, terminate")
            p.terminate()
            p.join()
        time.sleep(0.1)
#def iftttpush(temp,wind,hum,pplup,ppldown,crowd):
