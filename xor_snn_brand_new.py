import sys
puppet = int(sys.argv[1])
# coding: utf-8

# In[3]:

#w1_init = w1
#w2_init = w2
#w1=w1
#w2=w2
import numpy as np
import random
np.random.seed(1)
a = np.random.rand(400) * 0.2 - 0.1
i1 = np.concatenate((a[0:100],a[100:200],a[200:300] + 1, a[300:400] + 1), axis = 0)           #0,0,1,1
i2 = np.concatenate((a[0:100], a[100:200] + 1, a[200:300], a[300:400] + 1), axis = 0)         #0,1,0,1
#time = (np.arange(200) * 0.1).tolist()
max_i1 = np.max(i1)
max_i2 = np.max(i2)
min_i1 = np.min(i1)
min_i2 = np.min(i2)

#inputs = np.zeros((480*4, len(time)))
outputs = np.concatenate((np.zeros(100),np.ones(100),np.ones(100),np.zeros(100)),axis = 0)

temp_list = []

from random import shuffle
for i in range(400):
    temp_list.append((i1[i],i2[i], outputs[i]))
random.seed(puppet)
shuffle(temp_list)
for i in range(400):
    i1[i] = temp_list[i][0]
    i2[i] = temp_list[i][1]
    outputs[i] = temp_list[i][2]




# In[2]:

import numpy as np
encode_arr=np.zeros((400,12))
sigma=1/6.
for j in range(400):
    for i in range(6):
        c1=((2*i-3)*(max_i1-min_i1))/8
        c2 = ((2*i-3)*(max_i2-min_i2))/8
        encode_arr[j,i]=10-round(10*np.exp(-(i1[j]-c1)**2/(2*sigma*sigma)))
        encode_arr[j,i+6]=10-round(10*np.exp(-(i2[j]-c2)**2/(2*sigma*sigma)))
    
    print (j)
    


# In[28]:

#encode_arr.shape


# In[3]:

#np.max(i2)-np.min(i2)


# In[46]:

#outputs[159]==1


# In[4]:

import numpy as np

k=[1,5,9,13]
tau = 11.
t_i = [6,6,0, 0, 0, 6, 6, 0]
inputs = np.zeros((4800, 50))
#outputs = [0,0,1,1]
count=0
for i in range(400):
    
    for j in range(12):
    
    
#    for l,d in enumerate(k):
        for t in range(0,50):
            inputs[i*12+j,t] = ((t-encode_arr[i][j])/tau) * np.exp(1 - (t-encode_arr[i][j])/tau)
    print (i)
#    count=count+4
        
        


for i in range(0,4800):
    for t in range(0, 50):
        if inputs[i,t]<0.:
            inputs[i,t]=0.


# In[5]:

#inputs[0]


# In[6]:

#outputs


# In[7]:

import numpy as np
np.random.seed(100)
w1=np.random.uniform(0.0,0.3,(12,3))
np.random.seed(100)
w2=np.random.uniform(0.0,0.3,(3,1))

#w1=w1
#w2=w2

#w = w1
lr = 0.001


delta1=np.zeros((12,3))
delta2=np.zeros((3,1))
thr = 1.0
y = np.zeros((1,50))
for epochs in range(500):
    
    for i in range(400):
        y1 = np.dot(np.transpose(w1),inputs[12*i:12*(i+1),:])
    # print y1
        t_h = np.argmax(y1>=thr,axis=1)
        inputs_h = np.zeros((3, 50))
    
        count=0
        for x in range(3):
    
#            for l,d in enumerate(k):
            for t in range(0,50):
                inputs_h[x,t] = ((t-t_h[x])/tau) * np.exp(1 - (t-t_h[x])/tau)
            
#            count=count+4
        
        


        for x in range(0,3):
            for t in range(0, 50):
                if inputs_h[x,t]<0.:
                    inputs_h[x,t]=0.
        y2 = np.dot(np.transpose(w2),inputs_h)
    #print (w, o1, o2)
        t_s = np.argmax(y2>=thr,axis=1)
        o1=0.
    #  o2=0.
        for b in range(3):
#           for j in range(0,4):
            o1+=w2[b,0]*((t_s-t_h[b])/tau) * np.exp(1 - (t_s -t_h[b])/tau) * (1./(t_s-t_h[b]+1e-6) - 1./tau)
    #    for j in range(0,4):
     #       o2+=w[j+4,0]*((t_s-t_i[2*i +1]-k[j%4])/tau) * np.exp(1 - (t_s -t_i[2*i +1]-k[j%4])/tau) * (1./(t_s-t_i[2*i +1]-k[j%4]+1e-6 ) - 1./tau)
        if o1[0]<0.1:
            o1[0] = 0.1
        print (o1)
        if(outputs[i] == 0):
            print(epochs, t_s,t_s-16, i1[i],i2[i],'0')
            a11=t_s-16.
            a31=-1./o1
            a41=np.zeros((3,1))
            a51=np.zeros((3,1))
            a61=np.zeros((12,3))
            for b in range(3):
                for j in range(0,12):
                    a51[b,0]+=w1[j,b]*((t_h[b]-encode_arr[i][j])/tau) * np.exp(1 - (t_h[b] -encode_arr[i][j])/tau) * (1./(t_h[b]-encode_arr[i][j] +1e-6) - 1./tau)
#                for j in range(0,4):
#                    a51[b,0]+=w1[j+4,b]*((t_h[b]-t_i[2*i +1]-k[j%4])/tau) * np.exp(1 - (t_h[b] -t_i[2*i +1]-k[j%4])/tau) * (1./(t_h[b]-t_i[2*i +1]-k[j%4]+1e-6 ) - 1./tau)
            if a51[0][0]<0.1:
                a51[0][0]=0.1
            if a51[1][0]<0.1:
                a51[1][0]=0.1
            if a51[2][0]<0.1:
                a51[2][0]=0.1
            for b in range(3):
                for m in range(0,12):
            
                    a61[m,b]=((t_h[b]-encode_arr[i][m])/tau) * np.exp(1 - (t_h[b] -encode_arr[i][m])/tau)
#                for m in range(0,4):
            
#                    a61[m+4,b]=((t_h[b]-t_i[2*i +1]-k[m%4])/tau) * np.exp(1 - (t_h[b] -t_i[2*i +1]-k[m%4])/tau)
            
            
            
            for b in range(3):
#                for j in range(0,12):
                
                a41[b,0]+=w2[b,0]*((t_s-t_h[b])/tau) * np.exp(1 - (t_s -t_h[b])/tau) * (1./(t_s-t_h[b] +1e-6) - 1./tau)       
                
            matrix1=(a31)*(-a41)*a11
            delta1=np.dot(np.dot(a61,-1./a51),np.transpose(matrix1))
            for b in range(3):
#                for m in range(0,4):
            
                a21=((t_s-t_h[b])/tau) * np.exp(1 - (t_s -t_h[b])/tau)
                delta2[b,0]=a11*a31*a21
    #        for m in range(4):
    #         a21=((t_s-t_i[2*i +1]-k[m%4])/tau) * np.exp(1 - (t_s -t_i[2*i +1]-k[m%4])/tau) 
     #           delta[m+4,0]=a11*a31*a21
        elif(outputs[i]==1):
            print(epochs, t_s,t_s-10,i1[i],i2[i],'1')
            a11=t_s-10.
            a31=-1./o1
            a41=np.zeros((3,1))
            a51=np.zeros((3,1))
            a61=np.zeros((12,3))
            for b in range(3):
                for j in range(0,12):
                    a51[b,0]+=w1[j,b]*((t_h[b]-encode_arr[i][j])/tau) * np.exp(1 - (t_h[b] -encode_arr[i][j])/tau) * (1./(t_h[b]-encode_arr[i][j] +1e-6) - 1./tau)
#                for j in range(0,4):
#                    a51[b,0]+=w1[j+4,b]*((t_h[b]-t_i[2*i +1]-k[j%4])/tau) * np.exp(1 - (t_h[b] -t_i[2*i +1]-k[j%4])/tau) * (1./(t_h[b]-t_i[2*i +1]-k[j%4]+1e-6 ) - 1./tau)
            if a51[0][0]<0.1:
                a51[0][0]=0.1
            if a51[1][0]<0.1:
                a51[1][0]=0.1
            if a51[2][0]<0.1:
                a51[2][0]=0.1
            for b in range(3):
                for m in range(0,12):
            
                    a61[m,b]=((t_h[b]-encode_arr[i][m])/tau) * np.exp(1 - (t_h[b] -encode_arr[i][m])/tau)
#                for m in range(0,4):
            
#                    a61[m+4,b]=((t_h[b]-t_i[2*i +1]-k[m%4])/tau) * np.exp(1 - (t_h[b] -t_i[2*i +1]-k[m%4])/tau)
            
            
            
            for b in range(3):
#                for j in range(0,12):
                
                a41[b,0]+=w2[b,0]*((t_s-t_h[b])/tau) * np.exp(1 - (t_s -t_h[b])/tau) * (1./(t_s-t_h[b] +1e-6) - 1./tau)       
                
            matrix1=(a31)*(-a41)*a11
            delta1=np.dot(np.dot(a61,-1./a51),np.transpose(matrix1))
            for b in range(3):
#                for m in range(0,4):
            
                a21=((t_s-t_h[b])/tau) * np.exp(1 - (t_s -t_h[b])/tau)
                delta2[b,0]=a11*a31*a21
            
            
        #    a11=t_s-10.
         #   a31=-1./(o1+o2)
        #   for m in range(4):
        #        a21=((t_s-t_i[2*i]-k[m%4])/tau) * np.exp(1 - (t_s -t_i[2*i]-k[m%4])/tau)
         #       delta[m,0]=a11*a31*a21
        #   for m in range(4):
        #        a21=((t_s-t_i[2*i +1]-k[m%4])/tau) * np.exp(1 - (t_s -t_i[2*i +1]-k[m%4])/tau) 
         #       delta[m+4,0]=a11*a31*a21
           
        w2 = w2 - lr * delta2
        w1=w1-lr*delta1
        
        
        
        
        
        
        delta1=np.zeros((12,3))
        delta2=np.zeros((3,1))

test_arr=np.zeros((4,12))
t1=[0,1,1,0]
t2=[0,0,1,1]
for j in range(4):
    for i in range(6):
        c=(2*i-3)/8
        test_arr[j,i]=10-round(10*np.exp(-(t1[j]-c)**2/(2*sigma*sigma)))
        test_arr[j,i+6]=10-round(10*np.exp(-(t2[j]-c)**2/(2*sigma*sigma)))
    

tau = 11.
t_i = [6,6,0, 0, 0, 6, 6, 0]
tests = np.zeros((48, 50))  
for i in range(4):    
    for j in range(12):
    
    
#    for l,d in enumerate(k):
        for t in range(0,50):
            tests[i*12+j,t] = ((t-test_arr[i][j])/tau) * np.exp(1 - (t-test_arr[i][j])/tau)
    print (i)       
#    count=count+4
        
        


for i in range(0,48):
    for t in range(0, 50):
        if tests[i,t]<0.:
            tests[i,t]=0.

for i in range(4):
    y1 = np.dot(np.transpose(w1),tests[12*i:12*(i+1),:])
    # print y1
    t_h = np.argmax(y1>=thr,axis=1)
    inputs_h = np.zeros((3, 50))
    
    count=0
    for x in range(3):
    
#            for l,d in enumerate(k):
        for t in range(0,50):
            inputs_h[x,t] = ((t-t_h[x])/tau) * np.exp(1 - (t-t_h[x])/tau)
            
#            count=count+4
        
        


    for x in range(0,3):
        for t in range(0, 50):
            if inputs_h[x,t]<0.:
                inputs_h[x,t]=0.
    y2 = np.dot(np.transpose(w2),inputs_h)
    #print (w, o1, o2)
    t_s = np.argmax(y2>=thr,axis=1)
    print (t_s)

# In[59]:

#inputs


# In[53]:

#y1


# In[51]:

#o1


# In[1]:



# In[11]:

