import serial
ser = serial.Serial('/dev/ttyUSB0', 115200)
ser.flush()

def sendloc(line):
    location=open("location.txt","w+")
    location.write(str(line))
    location.close()
while 1: 
    if(ser.in_waiting >0):
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
        sendloc(line)