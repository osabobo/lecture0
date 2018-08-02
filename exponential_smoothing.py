# -*- coding: utf-8 -*-
time=[]
value=[]
forcast=[]
num=0
alpha=float(input("Enter the smoothing constant(between 0 and 1)"   ))
for x in range(1,3):
     num=eval(input("Enter the time series"      ))
     
     
     time.append(num)
     
     
for x in range(1,3):
    value1= float(input("Enter the values from stock"    ))
    value.append(value1[x])
     


forcast.append(value1[0])
for x in range(1,3):
    forcast.append[(alpha*value1[x-1]+(1-alpha)*value1[x-1])]
    print( time[x],value[x],forcast[x])
        


"""def fib():
    a,b = 1,1
    num=eval(input("Please input what Fib number you want to be calculated: "))
    num_int=int(num-2)
    for i in range (num_int):
       a,b = b,a+b
    print(b)

fib()"""



    
        



