import numpy as np
from PIL import Image
from  pylab import *
import time as tm
import matplotlib.pyplot as  plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
#CONVERTION FROM  DECIMAL NUMBER TO BINARY NUMBER
def DEC_BIN(n):
   b=np.zeros(8)
   i=7
   while(n>0):
       b[i]=n%2
       n=n//2
       i=i-1
   return(b)
#CONVERTION FROM  BINARY NUMBER TO DECIMAL NUMBER 
def BIN_DEC(x):
    s=0
    j=7
    for i in range(0,8):
        s=s+x[j]*2**i
        j=j-1
    return(s)
   
def encode(x1,x2,x3,x4,k):
    #print("orinal  values",x1,x2,x3,x4,k)
    y1=DEC_BIN(x1)
    y2=DEC_BIN(x2)
    y3=DEC_BIN(x3)
    y4=DEC_BIN(x4)
    k1=DEC_BIN(k)
    y1[7]=k1[0]
    y1[6]=k1[1]
    y2[7]=k1[2]
    y2[6]=k1[3]
    y3[7]=k1[4]
    y3[6]=k1[5]
    y4[7]=k1[6]
    y4[6]=k1[7]
    x1=int(BIN_DEC(y1))
    x2=int(BIN_DEC(y2))
    x3=int(BIN_DEC(y3))
    x4=int(BIN_DEC(y4))
    return(x1,x2,x3,x4)

    
def decode(x1,x2,x3,x4):
    k1=np.zeros(8)
    y1=DEC_BIN(x1)
    y2=DEC_BIN(x2)
    y3=DEC_BIN(x3)
    y4=DEC_BIN(x4)
    k1[0]=y1[7]
    k1[1]=y1[6]
    k1[2]=y2[7]
    k1[3]= y2[6]
    k1[4]= y3[7]
    k1[5]=y3[6]
    k1[6]=y4[7]
    k1[7]= y4[6]
    x1=int(BIN_DEC(y1))
    x2=int(BIN_DEC(y2))
    x3=int(BIN_DEC(y3))
    x4=int(BIN_DEC(y4))
    k=int(BIN_DEC(k1))
    return(k)
    
#MAIN PROGRAM     
if __name__ == "__main__":
    
    #read color image
    ii=Image.open('fepsi.jfif')
    #resize color image
    ii = ii.resize((80,80))
    #convert color image to gray image
    im=array(ii.convert('L'))
    m,n=im.shape
    A=im.flatten()

    #read color image
    jj=Image.open('sec.jfif')
    #resize color image
    jj = jj.resize((40,40))
    #convert color image to gray image
    im1=array(jj.convert('L'))
    m1,n1=im1.shape
    A1=im1.flatten()
    A2=np.zeros(m*n)
    k=0
    for i in range(0,(m*n),4):
       x=A[i:i+4]
       xx=A1[k]
       k=k+1
       #print(x)
       x1,x2,x3,x4=encode(x[0],x[1],x[2],x[3],xx)
       A2[i]=x1
       A2[i+1]=x2
       A2[i+2]=x3
       A2[i+3]=x4
    AA=np. reshape(A, (m, n))
    AA1=np. reshape(A1, (m1, n1))
    AA2=np. reshape(A2, (m, n))
    ''''print("logo",AA)
    print("secret",AA1)
    print("steg",AA2)'''
    k=0
    for i in range(0,(m*n),4):
       x=A2[i:i+4]
       #xx=A1[k]
       #k=k+1
       #print(x)
       xx=decode(x[0],x[1],x[2],x[3])
       A1[k]=xx
       k=k+1
    AAA=np. reshape(A, (m, n))
    AAA1=np. reshape(A1, (m1, n1))
    ''''print("logo",AAA)
    print("secret",AAA1)'''


    plt.subplot(3,3,1)
    plt.imshow(ii)
    plt.title("original")
    #plt.show()

    plt.subplot(3,3,2)
    plt.imshow(jj)
    plt.title("secrete")
    #plt.show()
    plt.subplot(3,3,3)
    AA=Image.fromarray(AA)
    plt.imshow(AA.convert('RGB'))
    plt.title("logo gray image  ")
    #plt.show()

    plt.subplot(3,3,4)
    AA1=Image.fromarray(AA1)
    plt.imshow(AA1.convert('RGB'))
    plt.title("secret gray  image  ")
    #plt.show()

    plt.subplot(3,3,5)
    AA2=Image.fromarray(AA2)
    plt.imshow(AA2.convert('RGB'))
    plt.title("steg Image")
    #plt.show()

    plt.subplot(3,3,6)
    AAA=Image.fromarray(AAA)
    plt.imshow(AAA.convert('RGB'))
    plt.title("logo Image")    
    #plt.show()

    plt.subplot(3,3,7)
    AAA1=Image.fromarray(AAA1)
    plt.imshow(AAA1.convert('RGB'))
    plt.title("secrete image ")
    plt.show()
    
    plt.title("histogram of original RED and encrypted BLUE Image")
    plt.xlabel("gray level values")
    plt.ylabel("gray level count")
    histogram, bin_edges = np.histogram(AA, bins=256, range=(0, 256))
    plt.plot( histogram,color="red")
    histogram1, bin_edges1= np.histogram(AA2, bins=256, range=(0, 256))
    plt.plot( histogram1, color="blue")
    histogram1, bin_edges1= np.histogram(AA1, bins=256, range=(0, 256))
    plt.plot( histogram1, color="green")
    plt.show()

    mse = np.mean((A - A2) ** 2)
    if mse==0:
       mse=1
    psnr = 20 * log10(255.0 / sqrt(mse))
    print("psnr(existing method)",psnr)

    '''im1=array(AA2.convert('L'))
    entr_img1 =np.sum(entropy(im1, disk(10)))/(m*n)
    im=array(ii.convert('L'))
    entr_img2 =np.sum(entropy(im, disk(10)))/(m*n)
    print('entropy',entr_img2-entr_img1)'''
  
