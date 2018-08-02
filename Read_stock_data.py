openamt=[]
highamt=[]
lowamt=[]

f1=open("C:\\Users\\Tolu\\Desktop\\file1.csv", 'r')

for s1 in f1:
    position_firstcomm=s1.find(",")
    position_secondcomm=s1.find(",", position_firstcomm + 1)
    position_thirdcomm=s1.find(",",position_secondcomm +1)
    position_forthcomm=s1.find(",",position_thirdcomm +1)
    position_fifthcomm=s1.find(",",position_forthcomm+1)
    openamt.append(str(s1[position_thirdcomm+1:position_forthcomm]))
    highamt.append(str(s1[position_forthcomm+1:position_fifthcomm]))
    lowamt.append(str(s1[position_fifthcomm+1:]))

f1.close()

for x in range(0,len(openamt)):
     print(openamt[x],highamt[x],lowamt[x])
    
                
    
