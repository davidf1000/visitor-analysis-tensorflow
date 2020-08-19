from gpiozero import CPUTemperature
import time

while (True):
	try:
		x = CPUTemperature() 
		print(x.temperature )
	except:
		print("error")
	time.sleep(2)
