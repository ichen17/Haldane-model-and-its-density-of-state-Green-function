# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def haldane_z(N,m,t1,t2,phi,Np):
    p=np.linspace(-pi/2,pi/2,Np+1)
    d1=m+2*t2*np.sin(phi)*np.sin(2*p)
    d2=-2*t1*np.cos(p)
    d3=-t1+p*0
    d4=2*t2*np.cos(p+phi)
    d5=2*t2*np.cos(p-phi)
    d6=2*t2*np.cos(phi)*np.cos(2*p)
    a=np.zeros((Np+1,2*N,2*N))
    for i in range(N):
        a[:,i,i]=d1[:]+d6[:]
        a[:,i+N,i+N]=-d1[:]+d6[:]
        a[:,i+N,i]=d2[:]
        a[:,i,i+N]=d2[:]
    for i in range(N-1):
        a[:,i+1,i]=d4[:]
        a[:,i,i+1]=d4[:]
        a[:,i+N+1,i+N]=d5[:]
        a[:,i+N,i+N+1]=d5[:]
        a[:,i+1,i+N]=d3[:]
        a[:,i+N,i+1]=d3[:]
    b=np.linalg.eigvals(a)
    return b

def haldane_a(N,m,t1,t2,phi,Np):
     p=np.linspace(-pi/6,pi/6,Np+1)
     d1=m
     d2=-t1*np.exp(0.+p*2j)
     d3=-t1*np.exp(0.-p*2j)
     d4=t2*np.exp(0.-phi*1j)
     d5=t2*np.exp(0.+phi*1j)
     d6=2*t2*np.cos(3*p)*np.exp(0.+phi*1j)
     d7=2*t2*np.cos(3*p)*np.exp(0.-phi*1j)
     d8=-t1*np.exp(0.+p*1j)
     d9=-t1*np.exp(0.-p*1j)
     a=np.zeros((Np+1,2*N,2*N), dtype=complex)
     for i in range(N):
        a[:,i,i]=d1
        a[:,i+N,i+N]=-d1
        a[:,i,i+N]=d2[:]
        a[:,i+N,i]=d3[:]
     for i in range(N-1):
        a[:,i,i+1]=d6[:]
        a[:,i+1,i]=d7[:]
        a[:,N+i,i+1+N]=d7[:]
        a[:,i+1+N,N+i]=d6[:]
        a[:,i+1,i+N]=d9[:]
        a[:,i,i+N+1]=d9[:]
        a[:,i+N,i+1]=d8[:]
        a[:,i+N+1,i]=d8[:]
     for i in range(N-2):
        a[:,i,i+2]=d4
        a[:,i+2,i]=d5
        a[:,i+N,i+2+N]=d5
        a[:,i+N+2,i+N]=d4
     b,c=np.linalg.eig(a)
     return b

def E_hald(m,t2,phi,N):
    k1=np.linspace(-pi/2,pi/2,N+1)
    k2=np.linspace(-pi,pi,N+1)
    E=np.zeros((N+1,2*N+2))
    for i in range(N+1):
       E[:,i]=2*t2*np.cos(phi)*(np.cos(2*k1)+2*np.cos(k2[i])*np.cos(k1))+np.sqrt(1+4*np.cos(k1)*np.cos(k2[i])+4*np.square(np.cos(k1))+np.square(m+2*t2*np.sin(phi)*(np.sin(2*k1)-2*np.cos(k2[i])*np.sin(k1))))
       E[:,i+N+1]=2*t2*np.cos(phi)*(np.cos(2*k1)+2*np.cos(k2[i])*np.cos(k1))-np.sqrt(1+4*np.cos(k1)*np.cos(k2[i])+4*np.square(np.cos(k1))+np.square(m+2*t2*np.sin(phi)*(np.sin(2*k1)-2*np.cos(k2[i])*np.sin(k1))))
    E=np.reshape(E,2*(N+1)**2)
    return E    
   
def g(d,ei,ef):
    N=d.shape[0]-1
    h=(ef-ei)/N
    return ig(d,N+1,h)
    
def ig(d,N,h):
    if N == 1:
        return 0
    elif N==0:
        return 0
    elif N == 2:
        return (d[0]+d[1])*h/2
    elif N == 3:
        return (d[0]+4*d[1]+d[2])*h/3
    elif N == 4:
        return (d[0]+3*d[1]+3*d[2]+d[3])*h*3/8
    elif (N+1)%2 == 0:
        return sum(ig(d[i:i+3],3,h) for i in range(0,N-1,2))
    elif (N+1)%2 == 1:
        return sum(ig(d[i:i+3],3,h) for i in range(0,N-3,2))+ig(d[-4:],4,h)
    else:
        return

def Dos2(f,ei,ef,Nw,N):
    E=((f-ei)/(ef-ei)*Nw).astype(int)
    E[E<0]=0
    E[E>Nw-1]=Nw-1
    D=np.bincount(E,minlength=Nw-1)*(Nw-1)*(2*pi/N)*(pi/N)/((ef-ei)*12*pi)
    t=g(D,ei,ef)
    D1=D/t
    return D1
     
def D22(f,ei,ef,Nw,N):
    E=((f-ei)/(ef-ei)*Nw).astype(int)
    E[E<0]=0
    E[E>Nw-1]=Nw-1
    D=np.bincount(E,minlength=Nw-1)*(Nw-1)*(2*pi/N)*(pi/N)/((ef-ei)*12*pi)
    return D
    
def D11(f,ei,ef,Nw,N,m):
    E=((f-ei)/(ef-ei)*Nw).astype(int)
    E[E<0]=0
    E[E>Nw-1]=Nw-1
    D=np.bincount(E,minlength=Nw-1)*(Nw-1)*(pi/N)/((ef-ei)*m*2*pi)   
    return D    
    
def Dos1(f,ei,ef,Nw,N):
    E=((f-ei)/(ef-ei)*Nw).astype(int)
    E[E<0]=0
    E[E>Nw-1]=Nw-1
    D=np.bincount(E,minlength=Nw-1)*(Nw-1)*(pi/N)/(ef-ei)   
    t=g(D,ei,ef)
    D1=D/t
    return D1

def D33(f,ei,ef,Nw,N,m):
    E=((f-ei)/(ef-ei)*Nw).astype(int)
    E[E<0]=0
    E[E>Nw-1]=Nw-1
    D=np.bincount(E,minlength=Nw-1)*(Nw-1)*(pi/(3*N))/((ef-ei)*m*2)   
    return D
    
def Dos3(f,ei,ef,Nw,N):
    E=((f-ei)/(ef-ei)*Nw).astype(int)
    E[E<0]=0
    E[E>Nw-1]=Nw-1
    D=np.bincount(E,minlength=Nw-1)*(Nw-1)*(pi/(3*N))/(ef-ei)   
    t=g(D,ei,ef)
    D1=D/t
    return D1
    
def u(d,tf,w,N,v):
    h=tf/N
    a=np.linspace(0,tf,2*N+1)
    return u_cal(d,a,w,N,h,tf,v)*np.exp(0+w*a*1j)
    
def u_cal(d,a,w,N,h,tf,v):
    U=np.zeros((2*N+1), dtype=complex)
    g=G(0,tf,N,w,d,v)
    U[0]=1
    k1=0
    k2=-h*(g[1]*U[0]+g[0]*(U[0]+h*k1/2))/4
    k3=-h*(g[1]*U[0]+g[0]*(U[0]+h*k2/2))/4
    U[1]=U[0]+(k1+k2)*h/4
    k4=-h*(g[2]*U[0]+4*g[1]*U[1]+g[0]*(U[0]+h*k3))/6
    U[2]=U[0]+(k1+2*k2+2*k3+k4)*h/6
    for i in range(1,N):
          a1=g[:2*i+1][::-1]*U[:2*i+1]
          a2=g[3:2*i+2][::-1]*U[:2*i-1]
          a3=g[2:2*i+3][::-1]*U[:2*i+1]          
          k1=-ig(a1,2*i+1,h/2)
          k2=-3*h*(g[3]*U[2*i-2]+3*g[2]*U[2*i-1]+3*g[1]*U[2*i]+g[0]*(U[2*i]+k1*h/2))/16-ig(a2,2*i-1,h/2)
          k3=-3*h*(g[3]*U[2*i-2]+3*g[2]*U[2*i-1]+3*g[1]*U[2*i]+g[0]*(U[2*i]+k2*h/2))/16-ig(a2,2*i-1,h/2)
          U[2*i+1]=U[2*i]+(k1+k2)*h/4
          k4=-h*(g[2]*U[2*i]+4*g[1]*U[2*i+1]+g[0]*(U[2*i]+h*k3))/6-ig(a3,2*i+1,h/2)
          U[2*i+2]=U[2*i]+h*(k1+2*k2+2*k3+k4)/6
    return U
        
     
def G(ti,tf,N,ed,d,v):
    a=np.linspace(ti,tf,2*N+1)
    b=s(-a+ti,d,v)*np.exp(0-ed*(a-ti)*1j)
    return b
    
def s(a,d,v):
    Nw=d.shape[0]-1
    h=(12)/Nw
    w=np.linspace(-6,6,Nw+1)
    b=np.zeros(a.shape[0], dtype=complex)
    for i in range(a.shape[0]):
      r=v*d*np.exp(0-a[i]*w*1j)
      b[i]=ig(r,Nw,h)
    return b

def J2(ei,ef,m,t2,phi,Nw,N,h):
    k1=np.linspace(-pi,pi,N+1)
    k2=np.linspace(-pi/2,pi/2,N+1)
    E=np.zeros((N+1,2*N+2))
    U=np.zeros((N+1,N+1))
    V=np.zeros((N+1,2*N+2))
    for j in range(N+1):
        E[:,j]=2*t2*np.cos(phi)*(np.cos(2*k1)+2*np.cos(k2[j])*np.cos(k1))+np.sqrt(1+4*np.cos(k1)*np.cos(k2[j])+4*np.square(np.cos(k1))+np.square(m+2*t2*np.sin(phi)*(np.sin(2*k1)-2*np.cos(k2[j])*np.sin(k1))))
        E[:,j+N+1]=2*t2*np.cos(phi)*(np.cos(2*k1)+2*np.cos(k2[j])*np.cos(k1))-np.sqrt(1+4*np.cos(k1)*np.cos(k2[j])+4*np.square(np.cos(k1))+np.square(m+2*t2*np.sin(phi)*(np.sin(2*k1)-2*np.cos(k2[j])*np.sin(k1))))
        U[:,j]=np.sqrt(1+4*np.cos(k1)*np.cos(k2[j])+4*np.square(np.cos(k1))+np.square(m+t2*np.sin(phi)*(np.sin(2*k1)-np.cos(k2[j])*np.sin(k1))))
    if h==0: 
     for i in range(N+1):
        V[:,i]=(1+4*np.cos(k1)*np.cos(k2[i])+4*np.square(np.cos(k1)))/(2*(U[:,i])*(U[:,i]-m-t2*np.sin(phi)*(np.sin(2*k1)-np.cos(k2[i])*np.sin(k1))))
        V[:,i+N+1]=(U[:,i]-m-t2*np.sin(phi)*(np.sin(2*k1)-np.cos(k2[i])*np.sin(k1)))/(2*U[:,i])
    elif h==1:
      for k in range(N+1):
        V[:,k]=(U[:,k]-m-t2*np.sin(phi)*(np.sin(2*k1)-np.cos(k2[k])*np.sin(k1)))/(2*U[:,k])
        V[:,k+N+1]=(1+4*np.cos(k1)*np.cos(k2[k])+4*np.square(np.cos(k1)))/(2*(U[:,k])*(U[:,k]-m-t2*np.sin(phi)*(np.sin(2*k1)-np.cos(k2[k])*np.sin(k1))))        
    a=np.reshape(E,2*(N+1)**2)
    b=np.reshape(V,2*(N+1)**2)
    E=((a-ei)/(ef-ei)*Nw).astype(int)
    E[E<0]=0
    E[E>Nw-1]=Nw-1
    D=np.bincount(E,weights=b,minlength=Nw-1)*(Nw-1)*(2*pi/N)*(pi/N)/((ef-ei)*12*pi)
    x=D22(a,ei,ef,Nw,N)
    t=g(x,ei,ef)
    D1=D/t
    return D1     
        
def J1(ei,ef,N,m,t1,t2,phi,Nw,Np,e): 
    p=np.linspace(-pi/2,pi/2,Np+1)
    d1=m+2*t2*np.sin(phi)*np.sin(2*p)
    d2=-2*t1*np.cos(p)
    d3=-t1+p*0
    d4=2*t2*np.cos(p+phi)
    d5=2*t2*np.cos(p-phi)
    d6=2*t2*np.cos(phi)*np.cos(2*p)
    a=np.zeros((Np+1,2*N,2*N))
    for i in range(N):
        a[:,i,i]=d1[:]+d6[:]
        a[:,i+N,i+N]=-d1[:]+d6[:]
        a[:,i+N,i]=d2[:]
        a[:,i,i+N]=d2[:]
    for i in range(N-1):
        a[:,i+1,i]=d4[:]
        a[:,i,i+1]=d4[:]
        a[:,i+N+1,i+N]=d5[:]
        a[:,i+N,i+N+1]=d5[:]
        a[:,i+1,i+N]=d3[:]
        a[:,i+N,i+1]=d3[:]
    b,c=np.linalg.eig(a)
    d=np.linalg.inv(c)       
    M=np.square(d)
    t=M[:,:,e]
    b=np.reshape(b,2*N*Np+2*N).real
    t=np.reshape(t,2*N*Np+2*N).real
    E=((b-ei)/(ef-ei)*Nw).astype(int)
    E[E<0]=0
    E[E>Nw-1]=Nw-1
    D=np.bincount(E,weights=t,minlength=Nw-1)*(Nw-1)*(pi/Np)/((ef-ei)*N*2*pi)
    x=D11(b,ei,ef,Nw,Np,N)
    t=g(x,ei,ef)
    D1=D/t
    return D1    
    
def J3(ei,ef,N,m,t1,t2,phi,Nw,Np,e):
    p=np.linspace(-pi/6,pi/6,Np+1)
    d1=m
    d2=-t1*np.exp(0.+p*2j)
    d3=-t1*np.exp(0.-p*2j)
    d4=t2*np.exp(0.-phi*1j)
    d5=t2*np.exp(0.+phi*1j)
    d6=2*t2*np.cos(3*p)*np.exp(0.+phi*1j)
    d7=2*t2*np.cos(3*p)*np.exp(0.-phi*1j)
    d8=-t1*np.exp(0.+p*1j)
    d9=-t1*np.exp(0.-p*1j)
    a=np.zeros((Np+1,2*N,2*N), dtype=complex)
    for i in range(N):
        a[:,i,i]=d1
        a[:,i+N,i+N]=-d1
        a[:,i,i+N]=d2[:]
        a[:,i+N,i]=d3[:]
    for i in range(N-1):
        a[:,i,i+1]=d6[:]
        a[:,i+1,i]=d7[:]
        a[:,N+i,i+1+N]=d7[:]
        a[:,i+1+N,N+i]=d6[:]
        a[:,i+1,i+N]=d9[:]
        a[:,i,i+N+1]=d9[:]
        a[:,i+N,i+1]=d8[:]
        a[:,i+N+1,i]=d8[:]
    for i in range(N-2):
        a[:,i,i+2]=d4
        a[:,i+2,i]=d5
        a[:,i+N,i+2+N]=d5
        a[:,i+N+2,i+N]=d4
    b,c=np.linalg.eig(a)
    d=np.abs(np.linalg.inv(c))       
    M=np.square(d)
    t=M[:,:,e]
    b=np.reshape(b,2*N*Np+2*N).real
    t=np.reshape(t,2*N*Np+2*N)
    E=((b-ei)/(ef-ei)*Nw).astype(int)
    E[E<0]=0
    E[E>Nw-1]=Nw-1
    D=np.bincount(E,weights=t,minlength=Nw-1)*(Nw-1)*(pi/(3*Np))/((ef-ei)*N*2)
    x=D33(b,ei,ef,Nw,Np,N)
    t=g(x,ei,ef)
    D1=D/t
    return D1 
    
def t(p,a):
    plt.plot(*sum([(p,e) for e in a.T], ()))

def gb(m,t2,phi,N):
    k2=np.linspace(0,pi,N+1)
    k1=np.linspace(0,pi/3,N+1)
    f=k1[::-1]
    b=k2[::-1]
    j=np.linspace(0,1,4*N+4)
    E=np.zeros((2,4*N+4))
    for i in range(N+1):
        E[0,i]=2*t2*np.cos(phi)*(np.cos(2*k1[i])+2*np.cos(k2[i])*np.cos(k1[i]))+np.sqrt(1+4*np.cos(k1[i])*np.cos(k2[i])+4*np.square(np.cos(k1[i]))+np.square(m+2*t2*np.sin(phi)*(np.sin(2*k1[i])-2*np.cos(k2[i])*np.sin(k1[i]))))
        E[1,i]=2*t2*np.cos(phi)*(np.cos(2*k1[i])+2*np.cos(k2[i])*np.cos(k1[i]))-np.sqrt(1+4*np.cos(k1[i])*np.cos(k2[i])+4*np.square(np.cos(k1[i]))+np.square(m+2*t2*np.sin(phi)*(np.sin(2*k1[i])-2*np.cos(k2[i])*np.sin(k1[i]))))
        E[0,i+N+1]=2*t2*np.cos(phi)*(np.cos(2*f[i])+2*np.cos(k2[N])*np.cos(f[i]))+np.sqrt(1+4*np.cos(f[i])*np.cos(k2[N])+4*np.square(np.cos(f[i]))+np.square(m+2*t2*np.sin(phi)*(np.sin(2*f[i])-2*np.cos(k2[N])*np.sin(f[i]))))
        E[1,i+N+1]=2*t2*np.cos(phi)*(np.cos(2*f[i])+2*np.cos(k2[N])*np.cos(f[i]))-np.sqrt(1+4*np.cos(f[i])*np.cos(k2[N])+4*np.square(np.cos(f[i]))+np.square(m+2*t2*np.sin(phi)*(np.sin(2*f[i])-2*np.cos(k2[N])*np.sin(f[i]))))
        E[0,i+2*N+2]=2*t2*np.cos(phi)*(np.cos(-2*k1[i])+2*np.cos(k2[N])*np.cos(-k1[i]))+np.sqrt(1+4*np.cos(-k1[i])*np.cos(k2[N])+4*np.square(np.cos(-k1[i]))+np.square(m+2*t2*np.sin(phi)*(np.sin(-2*k1[i])-2*np.cos(k2[N])*np.sin(-k1[i]))))
        E[1,i+2*N+2]=2*t2*np.cos(phi)*(np.cos(-2*k1[i])+2*np.cos(k2[N])*np.cos(-k1[i]))-np.sqrt(1+4*np.cos(-k1[i])*np.cos(k2[N])+4*np.square(np.cos(-k1[i]))+np.square(m+2*t2*np.sin(phi)*(np.sin(-2*k1[i])-2*np.cos(k2[N])*np.sin(-k1[i]))))
        E[0,i+3*N+3]=2*t2*np.cos(phi)*(np.cos(-2*f[i])+2*np.cos(b[i])*np.cos(-f[i]))+np.sqrt(1+4*np.cos(-f[i])*np.cos(b[i])+4*np.square(np.cos(-f[i]))+np.square(m+2*t2*np.sin(phi)*(np.sin(-2*f[i])-2*np.cos(b[i])*np.sin(-f[i]))))
        E[1,i+3*N+3]=2*t2*np.cos(phi)*(np.cos(-2*f[i])+2*np.cos(b[i])*np.cos(-f[i]))-np.sqrt(1+4*np.cos(-f[i])*np.cos(b[i])+4*np.square(np.cos(-f[i]))+np.square(m+2*t2*np.sin(phi)*(np.sin(-2*f[i])-2*np.cos(b[i])*np.sin(-f[i]))))
    plt.plot(j,E[0,:])
    plt.plot(j,E[1,:])
    return
    
def BC(m,t2,phi,N):
    k1=np.linspace(-pi/2,pi/2,N+1)
    k2=np.linspace(-pi,pi,N+1)
    h1=pi/N
    h2=2*pi/N
    a=np.zeros((N+1,N+1))
    b=np.zeros((N+1,N+1))
    c=np.zeros((N+1,N+1))
    d=np.zeros((N+1,N+1))
    e=np.zeros((N+1,N+1))
    j=np.zeros(N+1)
    for i in range(N+1):
        a[:,i]=-8*t2*np.cos(k1[:])*np.square(np.cos(k2[i]))*np.sin(k1[:])*np.sin(phi)
        b[:,i]=-t2*(6+16*np.cos(2*k1[:])+2*np.cos(4*k1[:])+np.sin(2*(k1[:]-k2[i]))+np.sin(2*(k1[:]+k2[i])))*np.sin(phi)
        c[:,i]=-2*np.sin(2*k2[i])*(m-t2*np.sin(phi))
        d[:,i]=-4*np.cos(k2[i])*(t2*np.cos(k1[:])*(5+np.cos(2*k1[:]))*np.sin(phi)-6*t2*np.cos(k1[:])*np.square(np.sin(k1[:]))*np.sin(phi)+np.sin(k1[:])*(m+2*t2*np.sin(phi)))
        e[:,i]=(1+4*np.cos(k1[:])*np.cos(k2[i])+4*np.square(np.cos(k1[:]))+np.square(m+2*t2*np.sin(phi)*(np.sin(2*k1)-2*np.cos(k2[i])*np.sin(k1))))**(3/2)
        f=(a[:,i]+b[:,i]+c[:,i]+d[:,i])/e[:,i]
        j[i]=ig(f,N,h1)
    k=ig(j,N,h2)/(2*pi*pi)
    return k
    
def uc1(N,t2,Ns,tf,v,e):   
    X=np.linspace(0,tf,2*N+1)
    Y=np.linspace(-pi/2,pi/2,Ns+1)
    Z=np.zeros((2*N+1,Ns+1))
    for i in range(Ns+1):
        s=-pi/2+i*pi/Ns
        J=J1(-5,5,50,1,1,t2,s,1002,8000,e)
        Z[:,i]=np.abs(u(J,tf,-3*t2*np.cos(s),N,v))
    plt.figure()    
    cp = plt.contourf(Y, X, Z,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x ($\theta$)')
    plt.ylabel('y ($\epsilon \tua$)')
    plt.show()
    
def uc2(N,t2,Ns,tf,v,e):   
    X=np.linspace(0,tf,2*N+1)
    Y=np.linspace(-pi/2,pi/2,Ns+1)
    Z=np.zeros((2*N+1,Ns+1))
    for i in range(Ns+1):
        s=-pi/2+i*pi/Ns
        J=J2(-5,5,1,t2,s,1002,8000,e)
        Z[:,i]=np.abs(u(J,tf,-3*t2*np.cos(s),N,v))
    plt.figure()    
    cp = plt.contourf(Y, X, Z,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x ($\theta $)')
    plt.ylabel('y ($\epsilon \tau $)')
    plt.show()  
    
def uc3(N,t2,Ns,tf,v,e):   
    X=np.linspace(0,tf,2*N+1)
    Y=np.linspace(-pi/2,pi/2,Ns+1)
    Z=np.zeros((2*N+1,Ns+1))
    for i in range(Ns+1):
        s=-pi/2+i*pi/Ns
        J=J3(-5,5,50,1,1,t2,s,502,8000,e)
        Z[:,i]=np.abs(u(J,tf,-3*t2*np.cos(pi/3),N,v))
    plt.figure()    
    cp = plt.contourf(Y, X, Z,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x ($\theta $)')
    plt.ylabel('y ($\epsilon \tau $)')
    plt.show()   
    
def ucm1(N,t2,Ns,tf,v,e):   
    X=np.linspace(0,tf,2*N+1)
    Y=np.linspace(-1,1,Ns+1)
    Z=np.zeros((2*N+1,Ns+1))
    for i in range(Ns+1):
        s=-1+i*2/Ns
        J=J1(-5,5,50,s,1,t2,pi/6,1002,8000,e)
        Z[:,i]=np.abs(u(J,tf,-3*t2*np.cos(pi/6),N,v))
    plt.figure()    
    cp = plt.contourf(Y, X, Z,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x ($\epslion \tua $)')
    plt.ylabel(' ($\theta $)')
    plt.show()

def ucm2(N,t2,Ns,tf,v,e):   
    X=np.linspace(0,tf,2*N+1)
    Y=np.linspace(-1,1,Ns+1)
    Z=np.zeros((2*N+1,Ns+1))
    for i in range(Ns+1):
        s=-1+i*2/Ns
        J=J2(-5,5,s,t2,pi/6,1002,8000,e)
        Z[:,i]=np.abs(u(J,tf,-3*t2*np.cos(pi/6),N,v))
    plt.figure()    
    cp = plt.contourf(Y, X, Z,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()    
    
def ucm3(N,t2,Ns,tf,v,e):   
    X=np.linspace(0,tf,2*N+1)
    Y=np.linspace(-1,1,Ns+1)
    Z=np.zeros((2*N+1,Ns+1))
    for i in range(Ns+1):
        s=-1+i*2/Ns
        J=J3(-5,5,50,s,1,t2,pi/6,1002,8000,e)
        Z[:,i]=np.abs(u(J,tf,-3*t2*np.cos(pi/6),N,v))
    plt.figure()    
    cp = plt.contourf(Y, X, Z,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()
   
def v(u,d,tf,T,e,f):
    N=u.shape[0]-1
    Nd=d.shape[0]-1
    hd=12/Nd
    w=np.linspace(-6,6,Nd+1)
    t=np.linspace(0,tf,N+1)
    y=np.zeros(Nd+1)
    h=tf/N
    v1=np.zeros(N+1)
    for i in range(N+1):
        if i == 0:
           r=0 
           s=0
           q=s
        elif i%2 == 1:
           r=u[i-1:i+1]
           a=t[i-1:i+1]
           s=igg(r,a,2,h,w)+q
        elif i%2 == 0:
           r=u[i-2:i+1]
           a=t[i-2:i+1]
           s=igg(r,a,3,h,w)+q
           q=s
        y=f*d*s*np.conjugate(s)/(1+np.exp((w-e)/T))
        v1[i]=ig(y,Nd+1,hd)
    return v1 
    
def igg(r,t,N,h,w):
    f=np.zeros(w.shape[0])
    for i in range (w.shape[0]):
        e=r*np.exp(-w[i]*t*1j)
        f[i]=ig(e,N,h)
    return f
        
def delta2(ei,ef,m,t2,phi,Nw,N,h):
    k1=np.linspace(-pi,pi,N+1)
    k2=np.linspace(-pi/2,pi/2,N+1)
    w=np.linspace(ei,ef,Nw+1)
    E1=np.zeros(N+1)
    E2=np.zeros(N+1)
    U=np.zeros((N+1,N+1))
    V1=np.zeros(N+1)
    V2=np.zeros(N+1)
    y1=np.zeros(N+1)
    y2=np.zeros(N+1)
    y3=np.zeros(N+1)
    y4=np.zeros(N+1)
    s=np.zeros(2*N+2)
    s1=np.zeros(2*N+2)
    z=np.zeros(Nw+1)
    z1=np.zeros(Nw+1)
    h1=2*pi/N
    h2=pi/N
    for e in range(Nw+1):
        for j in range(N+1):
          E1=2*t2*np.cos(phi)*(np.cos(2*k1)+2*np.cos(k2[j])*np.cos(k1))+np.sqrt(1+4*np.cos(k1)*np.cos(k2[j])+4*np.square(np.cos(k1))+np.square(m+2*t2*np.sin(phi)*(np.sin(2*k1)-2*np.cos(k2[j])*np.sin(k1))))
          E2=2*t2*np.cos(phi)*(np.cos(2*k1)+2*np.cos(k2[j])*np.cos(k1))-np.sqrt(1+4*np.cos(k1)*np.cos(k2[j])+4*np.square(np.cos(k1))+np.square(m+2*t2*np.sin(phi)*(np.sin(2*k1)-2*np.cos(k2[j])*np.sin(k1))))
          U[:,j]=np.sqrt(1+4*np.cos(k1)*np.cos(k2[j])+4*np.square(np.cos(k1))+np.square(m+t2*np.sin(phi)*(np.sin(2*k1)-np.cos(k2[j])*np.sin(k1))))
          if h==0: 
              V1=(1+4*np.cos(k1)*np.cos(k2[j])+4*np.square(np.cos(k1)))/(2*(U[:,j])*(U[:,j]-m-t2*np.sin(phi)*(np.sin(2*k1)-np.cos(k2[j])*np.sin(k1))))
              V2=(U[:,j]-m-t2*np.sin(phi)*(np.sin(2*k1)-np.cos(k2[j])*np.sin(k1)))/(2*U[:,j])
          elif h==1:
              V1=(U[:,j]-m-t2*np.sin(phi)*(np.sin(2*k1)-np.cos(k2[j])*np.sin(k1)))/(2*U[:,j])
              V2=(1+4*np.cos(k1)*np.cos(k2[j])+4*np.square(np.cos(k1)))/(2*(U[:,j])*(U[:,j]-m-t2*np.sin(phi)*(np.sin(2*k1)-np.cos(k2[j])*np.sin(k1)))) 
          y1=V1/(w[e]-E1+0.01j)
          y2=V2/(w[e]-E2+0.01j)
          y3=1/(w[e]-E1+0.01j)
          y4=1/(w[e]-E2+0.01j)
          s[j]=ig(y1,N+1,h1)
          s[j+N+1]=ig(y2,N+1,h1)
          s1[j]=ig(y3,N+1,h1).imag
          s1[j+N+1]=ig(y4,N+1,h1).imag
        z[e]=ig(s,2*N+1,h2)/(4*pi*pi) 
        z1[e]=ig(s1,2*N+1,h2)/(4*pi*pi)
    return z.real/g(z1,ei,ef)      

def delta1(ei,ef,N,m,t1,t2,phi,Nw,Np,e):
    p=np.linspace(-pi/2,pi/2,Np+1)
    d1=m+2*t2*np.sin(phi)*np.sin(2*p)
    d2=-2*t1*np.cos(p)
    d3=-t1+p*0
    d4=2*t2*np.cos(p+phi)
    d5=2*t2*np.cos(p-phi)
    d6=2*t2*np.cos(phi)*np.cos(2*p)
    a=np.zeros((Np+1,2*N,2*N))
    w=np.linspace(ei,ef,Nw+1)
    h1=pi/Np
    s=np.zeros(Nw+1)
    s1=np.zeros(Nw+1)
    z=np.zeros(N)
    z1=np.zeros(N)
    for i in range(N):
        a[:,i,i]=d1[:]+d6[:]
        a[:,i+N,i+N]=-d1[:]+d6[:]
        a[:,i+N,i]=d2[:]
        a[:,i,i+N]=d2[:]
    for i in range(N-1):
        a[:,i+1,i]=d4[:]
        a[:,i,i+1]=d4[:]
        a[:,i+N+1,i+N]=d5[:]
        a[:,i+N,i+N+1]=d5[:]
        a[:,i+1,i+N]=d3[:]
        a[:,i+N,i+1]=d3[:]
    b,c=np.linalg.eig(a)
    d=np.linalg.inv(c)       
    M=np.square(d)
    t=M[:,:,e]
    o=np.reshape(b,2*N*Np+2*N).real
    x=np.reshape(t,2*N*Np+2*N).real
    for q in range(Nw+1):
        for j in range(N):
            y=x[2*j*Np+2*j:2*(j+1)*Np+2*(j+1)]/(w[q]-o[2*j*Np+2*j:2*(j+1)*Np+2*(j+1)]-0.002j)
            y1=1/(w[q]-o[2*j*Np+2*j:2*(j+1)*Np+2*(j+1)]-0.002j)
            z[j]=ig(y,2*Np+2,h1)
            z1[j]=ig(y1,2*Np+2,h1).imag
        s1[q]= sum(z1)/(N*pi*2)   
        s[q]=sum(z)/(N*pi*2)
    return -s.real/g(s1,ei,ef)

def delta3(ei,ef,N,m,t1,t2,phi,Nw,Np,e):
    p=np.linspace(-pi/6,pi/6,Np+1)
    d1=m
    d2=-t1*np.exp(0.+p*2j)
    d3=-t1*np.exp(0.-p*2j)
    d4=t2*np.exp(0.-phi*1j)
    d5=t2*np.exp(0.+phi*1j)
    d6=2*t2*np.cos(3*p)*np.exp(0.+phi*1j)
    d7=2*t2*np.cos(3*p)*np.exp(0.-phi*1j)
    d8=-t1*np.exp(0.+p*1j)
    d9=-t1*np.exp(0.-p*1j)
    a=np.zeros((Np+1,2*N,2*N), dtype=complex)
    w=np.linspace(ei,ef,Nw+1)
    h1=pi/Np
    s=np.zeros(Nw+1)
    z=np.zeros(N)
    s1=np.zeros(Nw+1)
    z1=np.zeros(N)
    for i in range(N):
        a[:,i,i]=d1
        a[:,i+N,i+N]=-d1
        a[:,i,i+N]=d2[:]
        a[:,i+N,i]=d3[:]
    for i in range(N-1):
        a[:,i,i+1]=d6[:]
        a[:,i+1,i]=d7[:]
        a[:,N+i,i+1+N]=d7[:]
        a[:,i+1+N,N+i]=d6[:]
        a[:,i+1,i+N]=d9[:]
        a[:,i,i+N+1]=d9[:]
        a[:,i+N,i+1]=d8[:]
        a[:,i+N+1,i]=d8[:]
    for i in range(N-2):
        a[:,i,i+2]=d4
        a[:,i+2,i]=d5
        a[:,i+N,i+2+N]=d5
        a[:,i+N+2,i+N]=d4
    b,c=np.linalg.eig(a)
    d=np.abs(np.linalg.inv(c))        
    M=np.square(d)
    t=M[:,:,e]
    o=np.reshape(b,2*N*Np+2*N).real
    x=np.reshape(t,2*N*Np+2*N)
    for q in range(Nw+1):
        for j in range(N):
            y=x[2*j*Np+2*j:2*(j+1)*Np+2*(j+1)]/(w[q]-o[2*j*Np+2*j:2*(j+1)*Np+2*(j+1)]+0.002j)
            y1=1/(w[q]-o[2*j*Np+2*j:2*(j+1)*Np+2*(j+1)]-0.002j)
            z[j]=ig(y,2*Np+2,h1)
            z1[j]=ig(y1,2*Np+2,h1).imag
        s1[q]= sum(z1)/(N*pi*2)
        s[q]=sum(z)/(N*pi*2)
    return -s.real/g(s1,ei,ef)
    
def J31(ei,ef,N,m,t1,t2,Nw,Np,Nn,e):
    x = np.linspace(ei, ef, Nw-1)
    y = np.linspace(-pi/2, pi/2, Nn+1)
    z=np.zeros((Nw-1,Nn+1))
    y, x = np.meshgrid(y, x)
    for i in range (Nn+1):
        p=i*pi/Nn-pi/2
        z[:,i] = J1(ei,ef,N,m,t1,t2,p,Nw,Np,e)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(y, x, z, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_zlim(-0, 0.015)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Original Code')

def Jc1(ei,ef,N,m,t1,t2,Nw,Np,Nn,e):
    x = np.linspace(ei, ef, Nw-1)
    y = np.linspace(-pi/2, pi/2, Nn+1)
    z=np.zeros((Nw-1,Nn+1))
    for i in range (Nn+1):
        p=i*pi/Nn-pi/2
        z[:,i] = J1(ei,ef,N,m,t1,t2,p,Nw,Np,e)
    plt.figure()    
    cp = plt.contourf(y, x, z,100,cmap=plt.cm.YlGn)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x ($\theta $)')
    plt.ylabel('y ($\epsilon \tau $)')
    plt.show()

def f1(ei,ef,N,m,t1,t2,phi,Nw,Np,e,w,v):
    y=np.linspace(ef,ei,Nw-1)+w
    a=J1(ei,ef,N,m,t1,t2,phi,Nw,Np,e)*v
    b=delta1(ei,ef,N,m,t1,t2,phi,Nw-2,Np,e)*v
    c=b-y
    z=np.zeros(Nw-1)
    for i in range(Nw-2):
        if a[i] > 0.001 :
           z[i] = 0
        elif a[i] < 0.001 :
             if  np.sign(c[i]*c[i+1]) == -1 :
                 z[i]=(ef-ei)/((ef-ei)-(b[i]-b[i-1])*(Nw-1))
             elif np.sign(c[i]*c[i+1]) == 0 :
                 z[i]=(ef-ei)/((ef-ei)-(b[i]-b[i-1])*(Nw-1))
             elif np.sign(c[i]*c[i+1]) == 1 :
                 z[i]=0
    return sum(z) 
    
def x1(ei,ef,N,t1,t2,Nw,Np,e,v,m,n):
    q=np.linspace(-pi/2,pi/2,m+1)
    p=np.linspace(-1,1,n+1)
    z=np.zeros((m+1,n+1))
    for i in range (m+1):
        a=-pi/2+i*pi/m
        for j in range (n+1):
            b=-1+j*2/n
            z[i,j]=f1(ei,ef,N,b,t1,t2,a,Nw,Np,e,-3*t2*np.cos(a),v)
    plt.figure()    
    cp = plt.contourf(p, q, z,100,cmap=plt.cm.YlGn)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x ($\theta $)')
    plt.ylabel('y ($\epsilon \tau $)')
    plt.show()           

def f2(ei,ef,m,t2,phi,Nw,Np,h,w,v):
    y=np.linspace(ef,ei,Nw-1)+w
    a=J2(ei,ef,m,t2,phi,Nw,Np,h)
    b=delta2(ei,ef,m,t2,phi,Nw-2,Np,h)*v
    c=b-y
    z=np.zeros(Nw-1)
    for i in range(Nw-2):
        if a[i] > 0.003 :
           z[i] = 0
        elif a[i] < 0.003:
             if  np.sign(c[i]*c[i+1]) == -1 :
                 z[i]=(ef-ei)/((ef-ei)-(b[i]-b[i-1])*(Nw-1))
             elif np.sign(c[i]*c[i+1]) == 0 :
                 z[i]=(ef-ei)/((ef-ei)-(b[i]-b[i-1])*(Nw-1))
             elif np.sign(c[i]*c[i+1]) == 1 :
                 z[i]=0
    return sum(z)                
        
def ut1(N,t2,Ns,e):   
    X=np.linspace(-1,1,N+1)
    Y=np.linspace(-pi,pi,Ns+1)
    Z1=np.zeros((N+1,Ns+1))
    Z2=np.zeros((N+1,Ns+1))
    Z3=np.zeros((N+1,Ns+1))
    for i in range(Ns+1):
        for j in range(N+1):
            s=-pi+2*i*pi/Ns
            a=-1+j*2/N
            J=J1(-6,6,50,a,1,t2,s,2502,5000,e)
            b=u(J,140,-3*t2*np.cos(s)+0.6,2800,10)
            c=v(b,J,140,10,-3*t2*np.cos(s),10)
            Z1[j,i]=np.abs(b)[5600]
            Z2[j,i]=c[5600]
            Z3[j,i]=np.abs(b)[5600]**2+c[5600]
    plt.figure()    
    cp = plt.contourf(Y, X, Z1,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.xlabel('($\phi$)')
    plt.ylabel(' ($M$)')   
    
    plt.figure()
    cp = plt.contourf(Y, X, Z2,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.xlabel(' ($\phi$)')
    plt.ylabel(' ($M$)')
    
    plt.figure()
    cp = plt.contourf(Y, X, Z3,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.xlabel('($\phi$)')
    plt.ylabel(' ($M$)')
    plt.show()     

    
def ut2(N,t2,Ns,e):   
    X=np.linspace(-1,1,N+1)
    Y=np.linspace(-pi,pi,Ns+1)
    Z1=np.zeros((N+1,Ns+1))
    Z2=np.zeros((N+1,Ns+1))
    Z3=np.zeros((N+1,Ns+1))
    for i in range(Ns+1):
        for j in range(N+1):
            s=-pi+2*i*pi/Ns
            a=-1+j*2/N
            J=J2(-6,6,a,t2,s,2002,4000,e)
            b=u(J,140,-3*t2*np.cos(s),2800,0.2)
            c=v(b,J,140,10,-3*t2*np.cos(s),0.2)
            Z1[j,i]=np.abs(b)[5600]
            Z2[j,i]=c[5600]
            Z3[j,i]=np.abs(b)[5600]**2+c[5600]
            
    plt.figure()    
    cp = plt.contourf(Y, X, Z1,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.xlabel('$\phi$')
    plt.ylabel(' $M$')   
    
    plt.figure()
    cp = plt.contourf(Y, X, Z2,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.xlabel(' $\phi$')
    plt.ylabel(' $M$')
    
    plt.figure()
    cp = plt.contourf(Y, X, Z3,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.xlabel(' $\phi$')
    plt.ylabel(' $M$')
    plt.show()
    
def ut3(N,t2,Ns,e):   
    X=np.linspace(-1,1,N+1)
    Y=np.linspace(-pi,pi,Ns+1)
    Z1=np.zeros((N+1,Ns+1))
    Z2=np.zeros((N+1,Ns+1))
    Z3=np.zeros((N+1,Ns+1))
    for i in range(Ns+1):
        for j in range(N+1):
            s=-pi+2*i*pi/Ns
            a=-1+j*2/N
            J=J3(-6,6,50,a,1,t2,s,1502,3000,e)
            b=u(J,140,-3*t2*np.cos(s)+0.3,2800,10)
            c=v(b,J,140,10,-3*t2*np.cos(s),10)
            Z1[j,i]=np.abs(b)[5600]
            Z2[j,i]=c[5600]
            Z3[j,i]=np.abs(b)[5600]**2+c[5600]
    plt.figure()    
    cp = plt.contourf(Y, X, Z1,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.xlabel('$\phi$')
    plt.ylabel(' M')   
    
    plt.figure()
    cp = plt.contourf(Y, X, Z2,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.xlabel('$\phi$')
    plt.ylabel('M')
    
    plt.figure()
    cp = plt.contourf(Y, X, Z3,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.xlabel('$\phi$')
    plt.ylabel(' $M$')
    plt.show()
 
def z(c,n,ns):
    k1=np.linspace(-1,1,n+1)
    k2=np.linspace(-1,1,n+1)
    h=2/n
    g=np.zeros(2*n+2)
    g1=np.zeros(n+1)
    t=np.zeros(ns+1)
    for i in range(ns+1):
        for j in range(n+1):
            w=-4+8*i/ns
            g[:n+1]=1/(np.sqrt(k1**2+k2[j]**2)-w+0.001j)
            g[n+1:2*n+2]=1/(-np.sqrt(k1**2+k2[j]**2)-w+0.001j)
            g1[j]=ig(g[:n+1],n,h)+ig(g[n+1:2*n+2],n,h)
        t[i]=ig(g1,n,h)    
    return t.real

def w(q):
    a=np.linspace(-4,4,1001)
    e=q*a*np.log(np.abs((a-1)/(a)))-q*a*np.log(np.abs((a)/(a+1)))
    return e
  
 

def vc1(N,t2,Ns,tf,f,e):   
    X=np.linspace(0,tf,2*N+1)
    Y=np.linspace(-pi/2,pi/2,Ns+1)
    Z=np.zeros((2*N+1,Ns+1))
    for i in range(Ns+1):
        s=-pi/2+i*pi/Ns
        J=J1(-5,5,50,1,1,t2,s,1002,8000,e)
        a=u(J,tf,-3*t2*np.cos(s),N,f)
        Z[:,i]=v(a,J,tf,10,-3*t2*np.cos(s),f)
    plt.figure()    
    cp = plt.contourf(Y, X, Z,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x ($\phi$)')
    plt.ylabel('y ($\epsilon \tau$)')
    plt.show()      
  
def vt1(N,t2,Ns,e):   
    X=np.linspace(-1,1,N+1)
    Y=np.linspace(-pi/2,pi/2,Ns+1)
    Z=np.zeros((N+1,Ns+1),dtype=complex)
    for i in range(Ns+1):
        for j in range(N+1):
            s=-pi/2+i*pi/Ns
            a=-1+j*2/N
            J=J1(-6,6,50,a,1,t2,s,4002,4000,e)
            b=u(J,140,-3*t2*np.cos(s),2800,10)
            Z[j,i]=v(b,J,140,0,-3*t2*np.cos(s),10)[5600]
    plt.figure()    
    cp = plt.contourf(Y, X, Z,100,cmap=plt.cm.hot)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x ($ \phi $)')
    plt.ylabel('y ($\epsilon \tau$)')
    plt.show()   
    
def Gra(ei,ef,Nw,Np,h):
    f=J2(ei,ef,0,0,0,Nw,Np,h)
    c=f.shape[0]
    if c%2 == 0:
        f[c/2]=0
        f[c/2+1]=0
    if c%2 == 1:
        f[(c+1)/2]=0
    return f
    
def U1(ei,ef,N,m,t1,t2,phi,Nw,Np,e,w,v):
    r=np.linspace(ei,ef,Nw+1)
    a=J1(ei,ef,N,m,t1,t2,phi,Nw+2,Np,e)*v
    b=delta1(ei,ef,N,m,t1,t2,phi,Nw,Np,e)*v
    h=0.0000001
    c=1/(r-(a+h)*1j-b-w)
    return c.imag
    
def U2(ei,ef,m,t2,phi,Nw,Np,h,w,v):
    r=np.linspace(ei,ef,Nw+1)
    a=J2(ei,ef,m,t2,phi,Nw+2,Np,h)*v
    b=delta2(ei,ef,m,t2,phi,Nw,Np,h)*v
    h=0.000001
    c=1/(r-(a+h)*1j-b-w)
    return c.imag

def U4(ei,ef,Nw,Np,h,w,v):
    r=np.linspace(ei,ef,Nw+1)
    a=Gra(ei,ef,Nw+2,Np,h)*v
    b=delta2(ei,ef,0,0,0,Nw,Np,h)*v
    h=0.000001
    c=1/(r-(a+h)*1j-b-w)
    return c.imag
    
def U3(ei,ef,N,m,t1,t2,phi,Nw,Np,e,w,v):
    r=np.linspace(ei,ef,Nw+1)
    a=J3(ei,ef,N,m,t1,t2,phi,Nw+2,Np,e)*v
    b=delta3(ei,ef,N,m,t1,t2,phi,Nw,Np,e)*v
    h=0.00001
    c=1/(r-(a+h)*1j-b-w)
    return c.imag

def U1P(ei,ef,Nw,Np,e):
    a=U1(ei,ef,50,0.5,1,1/6,pi/3,Nw,Np,e,-np.cos(pi/3)/2,10)
    b=U1(ei,ef,50,0.5,1,1/6,pi/6,Nw,Np,e,-np.cos(pi/6)/2,10)
    c=U1(ei,ef,50,0,1,0,0,Nw,Np,e,0,10)
    d=np.linspace(ei,ef,Nw+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_c = r"$\phi=0 m=0 t_{2}=0$"
    label_b = r"$\phi=\pi/3 m=0.5 t_{2}=1/6$"
    label_a = r"$\phi=\pi/6 m=0.5 t_{2}=1/6$"
    plt.plot(d,c,"g",label=label_c,linewidth=0.7)
    plt.plot(d,b,"b",label=label_b,linewidth=0.7)
    plt.plot(d,a,"r",label=label_a,linewidth=0.7)
    plt.axis([ei,ef,0.0,2.0])
    plt.xlabel('Energy ($ \epsilon $)',fontsize=14)
    plt.ylabel('$A$',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def U2P(ei,ef,Nw,Np,h):
    a=U2(ei,ef,0.5,1/6,pi/3,Nw,Np,h,-np.cos(pi/3)/2,0.2)
    b=U2(ei,ef,0.5,1/6,pi/6,Nw,Np,h,-np.cos(pi/6)/2,0.2)
    c=U4(ei,ef,Nw,Np,h,0,0.2)
    d=np.linspace(ei,ef,Nw+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_c = r"$\phi=0 m=0 t_{2}=0$"
    label_b = r"$\phi=\pi/3 m=0.5 t_{2}=1/6$"
    label_a = r"$\phi=\pi/6 m=0.5 t_{2}=1/6$"
    plt.plot(d,c,"g",label=label_c,linewidth=0.7)
    plt.plot(d,b,"b",label=label_b,linewidth=0.7)
    plt.plot(d,a,"r",label=label_a,linewidth=0.7)
    plt.axis([ei,ef,0.0,1.0])
    plt.xlabel('Energy ($ \epsilon $)',fontsize=14)
    plt.ylabel('$A$',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def U3P(ei,ef,Nw,Np,h):
    a=U3(ei,ef,50,0.5,1,1/6,pi/3,Nw,Np,h,-np.cos(pi/3)/2,10)
    b=U3(ei,ef,50,0.5,1,1/6,pi/6,Nw,Np,h,-np.cos(pi/6)/2,10)
    c=U3(ei,ef,50,0,1,0,0,Nw,Np,h,0,10)
    d=np.linspace(ei,ef,Nw+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_c = r"$\phi=0 m=0 t_{2}=0$"
    label_b = r"$\phi=\pi/3 m=0.5 t_{2}=1/6$"
    label_a = r"$\phi=\pi/6 m=0.5 t_{2}=1/6$"
    plt.plot(d,c,"g",label=label_c,linewidth=0.7)
    plt.plot(d,b,"b",label=label_b,linewidth=0.7)
    plt.plot(d,a,"r",label=label_a,linewidth=0.7)
    plt.axis([ei,ef,0.0,1.0])
    plt.xlabel('Energy ($ \epsilon $)',fontsize=14)
    plt.ylabel('$A$',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def Dd1(ei,ef,N):
    a=haldane_z(50,0.5,1,1/6,pi/3,2000)
    b=haldane_z(50,0.5,1,1/6,pi/6,2000)
    c=haldane_z(50,0,1,0,0,2000)
    a=np.reshape(a,200100)
    b=np.reshape(b,200100)
    c=np.reshape(c,200100)
    a=Dos1(a,ei,ef,N+2,2000)
    b=Dos1(b,ei,ef,N+2,2000)
    c=Dos1(c,ei,ef,N+2,2000)
    d=np.linspace(ei,ef,N+1)
    plt.figure(figsize=(12,6),dpi=98)
    p1=plt.subplot(121)
    p2=plt.subplot(122)
    label_c = r"Normal Graphene"
    label_b = r"$\phi=\pi/3 m=0.5 t_{2}=1/6$"
    label_a = r"$\phi=\pi/6 m=0.5 t_{2}=1/6$"
    p1.plot(d,c,"b",label=label_c,linewidth=0.7)
    p1.plot(d,b,"g",label=label_b,linewidth=0.7)
    p1.plot(d,a,"r",label=label_a,linewidth=0.7)
    p2.plot(d,c,"b",linewidth=1)
    p2.plot(d,b,"g",linewidth=1)
    p2.plot(d,a,"r",linewidth=1)
    p1.axis([ei,ef,0.0,1.4])
    p2.axis([-1,1,0.0,0.3])
    p1.set_xlabel('Energy ($ \epsilon $)',fontsize=14)
    p1.set_ylabel('DoS',fontsize=14)
    p2.set_xlabel('Energy ($ \epsilon $)',fontsize=14)
    p2.set_ylabel('DoS',fontsize=14)
    p1.grid(True)
    p1.legend()
    p2.grid(True)
    p2.legend()
    plt.show()

def Dd2(ei,ef,N):
    a=E_hald(0.5,1/6,pi/3,2000)
    b=E_hald(0.5,1/6,pi/6,2000)
    c=E_hald(0,0,0,2000)
    a=Dos2(a,ei,ef,N+2,2000)
    b=Dos2(b,ei,ef,N+2,2000)
    c=Dos2(c,ei,ef,N+2,2000)
    d=np.linspace(ei,ef,N+1)
    plt.figure(figsize=(12,6),dpi=98)
    p1=plt.subplot(121)
    p2=plt.subplot(122)
    label_c = r"Normal Graphene"
    label_b = r"$\phi=\pi/3  m=0.5  t_{2}=1/6$"
    label_a = r"$\phi=\pi/6  m=0.5  t_{2}=1/6$"
    p1.plot(d,c,"b",label=label_c,linewidth=1)
    p1.plot(d,b,"g",label=label_b,linewidth=1)
    p1.plot(d,a,"r",label=label_a,linewidth=1)
    p2.plot(d,c,"b",linewidth=1)
    p2.plot(d,b,"g",linewidth=1)
    p2.plot(d,a,"r",linewidth=1)
    p1.axis([ei,ef,0.0,1.4])
    p2.axis([-1,1,0.0,0.3])
    p1.set_xlabel('Energy ($ \epsilon $)',fontsize=14)
    p1.set_ylabel('DoS',fontsize=14)
    p2.set_xlabel('Energy ($ \epsilon $)',fontsize=14)
    p2.set_ylabel('DoS',fontsize=14)
    p1.grid(True)
    p1.legend()
    p2.grid(True)
    p2.legend()
    plt.show()    
    
def Dd3(ei,ef,N):
    a=haldane_a(50,0.5,1,1/6,pi/3,2000)
    b=haldane_a(50,0.5,1,1/6,pi/6,2000)
    c=haldane_a(50,0,1,0,0,2000)
    a=np.reshape(a,200100)
    b=np.reshape(b,200100)
    c=np.reshape(c,200100)
    a=Dos3(a,ei,ef,N+2,2000)
    b=Dos3(b,ei,ef,N+2,2000)
    c=Dos3(c,ei,ef,N+2,2000)
    d=np.linspace(ei,ef,N+1)
    plt.figure(figsize=(12,6),dpi=98)
    p1=plt.subplot(121)
    p2=plt.subplot(122)
    label_c = r"Normal Graphene"
    label_b = r"$\phi=\pi/3  m=0.5  t_{2}=1/6$"
    label_a = r"$\phi=\pi/6  m=0.5  t_{2}=1/6$"
    p1.plot(d,c,"b",label=label_c,linewidth=0.7)
    p1.plot(d,b,"g",label=label_b,linewidth=0.7)
    p1.plot(d,a,"r",label=label_a,linewidth=0.7)
    p2.plot(d,c,"b",linewidth=1)
    p2.plot(d,b,"g",linewidth=1)
    p2.plot(d,a,"r",linewidth=1)
    p1.axis([ei,ef,0.0,1.4])
    p2.axis([-1,1,0.0,0.5])
    p1.set_xlabel('Energy ($ \epsilon $)',fontsize=14)
    p1.set_ylabel('DoS',fontsize=14)
    p2.set_xlabel('Energy ($ \epsilon $)',fontsize=14)
    p2.set_ylabel('DoS',fontsize=14)
    p1.grid(True)
    p1.legend()
    p2.grid(True)
    p2.legend()
    plt.show()  
    
def J_1(ei,ef,N,e):
    a=J1(ei,ef,50,0.5,1,1/6,-pi/3,N+2,2000,e)
    b=J1(ei,ef,50,0.5,1,1/6,-pi/6,N+2,2000,e)
    c=J1(ei,ef,50,0,1,0,0,N+2,2000,e)
    d=np.linspace(ei,ef,N+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_c = r"\phi=0 m=0 t_{2}=0/"
    label_b = r"$\phi=\pi/3 m=0.5 t_{2}=1/6$"
    label_a = r"$\phi=\pi/6 m=0.5 t_{2}=1/6$"
    plt.plot(d,c,"g--",label=label_c,linewidth=1)
    plt.plot(d,b,"b:",label=label_b,linewidth=1)
    plt.plot(d,a,"r",label=label_a,linewidth=0.5)
    plt.axis([ei,ef,0.0,0.01])
    plt.xlabel('Energy ($ \epsilon $)',fontsize=14)
    plt.ylabel('$J(\epsilon)$',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def J_2(ei,ef,N,e):
    a=J2(ei,ef,0.5,1/6,pi/3,N+2,2000,e)
    b=J2(ei,ef,0.5,1/6,pi/6,N+2,2000,e)
    c=J2(ei,ef,0,0,0,N+2,2000,e)
    d=np.linspace(ei,ef,N+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_c = r"Normal Graphene"
    label_b = r"$\phi=\pi/3 m=0.5 t_{2}=1/6$"
    label_a = r"$\phi=\pi/6 m=0.5 t_{2}=1/6$"
    plt.plot(d,c,"b",label=label_c,linewidth=0.7)
    plt.plot(d,b,"g",label=label_b,linewidth=0.7)
    plt.plot(d,a,"r",label=label_a,linewidth=0.7)
    plt.axis([ei,ef,0.0,0.5])
    plt.xlabel('Energy ($ \epsilon $)',fontsize=14)
    plt.ylabel('$J(\epsilon)$',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def J_3(ei,ef,N,e):
    a=J3(ei,ef,50,0.5,1,1/6,pi/3,N+2,2000,e)
    b=J3(ei,ef,50,0.5,1,1/6,pi/6,N+2,2000,e)
    c=J3(ei,ef,50,0,1,0,0,N+2,2000,e)
    d=np.linspace(ei,ef,N+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_c = r"Normal Graphene"
    label_b = r"$\phi=\pi/3 m=0.5 t_{2}=1/6$"
    label_a = r"$\phi=\pi/6 m=0.5 t_{2}=1/6$"
    plt.plot(d,c,"b",label=label_c,linewidth=0.7)
    plt.plot(d,b,"g",label=label_b,linewidth=0.7)
    plt.plot(d,a,"r",label=label_a,linewidth=0.7)
    plt.axis([ei,ef,0.0,0.01])
    plt.xlabel('Energy ($ \epsilon $)',fontsize=14)
    plt.ylabel('$J(\epsilon)$',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()    
    
def A_1(ei,ef,Nn,m,t2,phi,N,Np,e,v):
    a=v*J1(ei,ef,Nn,m,1,t2,phi,N+2,Np,e)
    b=v*delta1(ei,ef,Nn,m,1,t2,phi,N,Np,e)
    c=np.linspace(ei,ef,N+1)
    d=np.linspace(ei,ef,N+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_a = r"Imaginay part self enegy"
    label_b = r"Real part self enegy"
    c=c+3*t2*np.cos(phi)
    plt.plot(d,a,"b",label=label_a,linewidth=0.7)
    plt.plot(d,b,"r",label=label_b,linewidth=0.7)
    plt.plot(d,c,"g",linewidth=0.7)
    plt.axis([ei,ef,-0.07,0.07])
    plt.xlabel('Energy ($ \epsilon $)',fontsize=14)
    plt.ylabel('$J(\epsilon)$',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def A_2(ei,ef,m,t2,phi,N,Np,e,v):
    a=v*J2(ei,ef,m,t2,phi,N+2,Np,e)
    b=v*delta2(ei,ef,m,t2,phi,N,Np,e)
    c=np.linspace(ei,ef,N+1)
    d=np.linspace(ei,ef,N+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_a = r"Imaginay part self enegy"
    label_b = r"Real part self enegy"
    c=c+3*t2*np.cos(phi)
    plt.plot(d,a,"b",label=label_a,linewidth=0.7)
    plt.plot(d,b,"r",label=label_b,linewidth=0.7)
    plt.plot(d,c,"g",linewidth=0.7)
    plt.axis([ei,ef,-0.1,0.1])
    plt.xlabel('Energy ($ \epsilon $)',fontsize=14)
    plt.ylabel('$J(\epsilon)$',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def A_3(ei,ef,Nn,m,t2,phi,N,Np,e,v):
    a=v*J3(ei,ef,Nn,m,1,t2,phi,N+2,Np,e)
    b=v*delta3(ei,ef,Nn,m,1,t2,phi,N,Np,e)
    c=np.linspace(ei,ef,N+1)
    d=np.linspace(ei,ef,N+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_a = r"Imaginay part self enegy"
    label_b = r"Real part self enegy"
    c=c+3*t2*np.cos(phi)
    plt.plot(d,a,"b",label=label_a,linewidth=0.7)
    plt.plot(d,b,"r",label=label_b,linewidth=0.7)
    plt.plot(d,c,"g",linewidth=0.7)
    plt.axis([ei,ef,-0.1,0.1])
    plt.xlabel('Energy ($ \epsilon $)',fontsize=14)
    plt.ylabel('$J(\epsilon)$',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def B_1(N,m,t1,t2,phi,Np):
    a=haldane_z(N,m,t1,t2,phi,Np)
    a=np.sort(a,axis=1)
    p=np.linspace(-pi/2,pi/2,Np+1)
    plt.figure(figsize=(8,6),dpi=98)
    t(p,a)
    plt.xlabel('k',fontsize=14)
    plt.ylabel('Energy ',fontsize=14)
    plt.axis([-pi/2,pi/2,-3.5,4])
    plt.grid(True)
    plt.legend()
    plt.show()
    
def B_3(N,m,t1,t2,phi,Np):
    a=haldane_a(N,m,t1,t2,phi,Np)
    a=np.sort(a,axis=1)
    p=np.linspace(-pi/2,pi/2,Np+1)
    plt.figure(figsize=(8,6),dpi=98)
    t(p,a)
    plt.xlabel('k',fontsize=14)
    plt.ylabel('Energy ',fontsize=14)
    plt.axis([-pi/2,pi/2,-3.5,4])
    plt.grid(True)
    plt.legend()
    plt.show()
 
def C_1(tf,N,e,v):
    a=J1(-6,6,50,0.5,1,1/6,pi/3,2002,4000,e)
    b=J1(-6,6,50,0.5,1,1/6,pi/6,2002,4000,e)
    c=J1(-6,6,50,0,1,0,0,2002,4000,e)
    a=np.abs(u(a,tf,-0.5*np.cos(pi/3),N,v))
    b=np.abs(u(b,tf,-0.5*np.cos(pi/6),N,v))
    c=np.abs(u(c,tf,0,N,v))
    d=np.linspace(0,tf,2*N+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_c = r"$\phi=0 m=0 t_{2}=0$"
    label_b = r"$\phi=\pi/3 m=0.5 t_{2}=1/6$"
    label_a = r"$\phi=\pi/6 m=0.5 t_{2}=1/6$"
    plt.plot(d,c,"b",label=label_c,linewidth=1)
    plt.plot(d,b,"g",label=label_b,linewidth=1)
    plt.plot(d,a,"r",label=label_a,linewidth=1)
    plt.axis([0,tf,0.0,1.1])
    plt.xlabel('time ($ \epsilon \tua $)',fontsize=14)
    plt.ylabel('|u|',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def C_2(tf,N,e,v):
    a=J2(-6,6,0.5,1/6,pi/6,4002,8000,e)
    b=J2(-6,6,0.5,1/6,pi/3,4002,8000,e)
    c=Gra(-6,6,4002,8000,e)
    a=np.abs(u(a,tf,-0.5*np.cos(pi/6),N,v))
    b=np.abs(u(b,tf,-0.5*np.cos(pi/3),N,v))
    c=np.abs(u(c,tf,0,N,v))
    d=np.linspace(0,tf,2*N+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_c = r"Normal Graphene"
    label_b = r"$\phi=\pi/3 m=0.5 t_{2}=1/6$"
    label_a = r"$\phi=\pi/6 m=0.5 t_{2}=1/6$"
    plt.plot(d,c,"b",label=label_c,linewidth=1)
    plt.plot(d,b,"g",label=label_b,linewidth=1)
    plt.plot(d,a,"r",label=label_a,linewidth=1)
    plt.axis([0,tf,0.4,1.1])
    plt.xlabel('time ($ \epsilon\tau $)',fontsize=14)
    plt.ylabel('u',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def C_3(tf,N,e,v):
    a=J3(-6,6,50,0.5,1,1/6,pi/6,4002,8000,e)
    b=J3(-6,6,50,0.5,1,1/6,pi/3,4002,8000,e)
    c=J3(-6,6,50,0,1,0,0,4002,8000,e)
    a=np.abs(u(a,tf,-0.5*np.cos(pi/6),N,v))
    b=np.abs(u(b,tf,-0.5*np.cos(pi/3),N,v))
    c=np.abs(u(c,tf,0,N,v))
    d=np.linspace(0,tf,2*N+1)
    plt.figure(figsize=(8,6),dpi=98)
    label_c = r"Normal Graphene"
    label_b = r"$\phi=\pi/3 m=0.5 t_{2}=1/6$"
    label_a = r"$\phi=\pi/6 m=0.5 t_{2}=1/6$"
    plt.plot(d,c,"b",label=label_c,linewidth=1)
    plt.plot(d,b,"g",label=label_b,linewidth=1)
    plt.plot(d,a,"r",label=label_a,linewidth=1)
    plt.axis([0,tf,0.0,1.1])
    plt.xlabel('time ($ \epsilon\tau $)',fontsize=14)
    plt.ylabel('u',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def Hs(N,t,J,phi):
    a=np.zeros((N*N,N*N))
    for i in range(N*N):
        a[i,i]=t
        a[i+N,i+N]=-d1[:]+d6[:]
        a[i+N,i]=d2[:]
        a[i,i+N]=d2[:]
    for i in range(N-1):
        a[i+1,i]=d4[:]
        a[i,i+1]=d4[:]
        a[i+N+1,i+N]=d5[:]
        a[i+N,i+N+1]=d5[:]
        a[i+1,i+N]=d3[:]
        a[i+N,i+1]=d3[:]
    b=np.linalg.eigvals(a)
    return b

def P_1(tf,N,v):
    for i in range(5):
        a=J1(-5,5,50,0.5,1,1/6,pi/3,1002,2000,i)
        a=np.abs(u(a,tf,-0.5*np.cos(pi/3),N,v))
        d=np.linspace(0,tf,2*N+1)
        plt.plot(d,a,linewidth=1)
    plt.axis([0,tf,0.0,1.0])
    plt.xlabel('time ($ \epsilon$ $\tau $)',fontsize=14)
    plt.ylabel('|u|',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def U12(ei,ef,m,t2,phi,Nw,Np,h,w,v):
    y=np.linspace(ef,ei,Nw-1)+w-(1/Nw)*1j/10
    b=delta2(ei,ef,m,t2,phi,Nw-2,Np,h)*v
    d=1/(y-b)
    p=np.linspace(ei,ef,d.shape[0])
    plt.figure(figsize=(8,6),dpi=98)
    plt.plot(p,d.imag,linewidth=1)
    plt.axis([ei,ef,0.0,0.05])
    plt.xlabel(' ($ \epsilon$)',fontsize=14)
    plt.ylabel('A',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

def U11(ei,ef,N,m,t2,phi,Nw,Np,h,w,v):
    y=np.linspace(ef,ei,Nw-1)+w-(1/Nw)*1j/10
    b=delta1(ei,ef,m,t2,phi,Nw-2,Np,h)*v
    d=1/(y-b)
    p=np.linspace(ei,ef,d.shape[0])
    plt.figure(figsize=(8,6),dpi=98)
    plt.plot(p,d.imag,linewidth=1)
    plt.axis([ei,ef,0.0,0.05])
    plt.xlabel(' ($ \epsilon$)',fontsize=14)
    plt.ylabel('A',fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

def P1(ei,ef,N,m,t1,t2,phi,Nw,Np,e):
    p=np.linspace(-pi/2,pi/2,Np+1)
    d1=m+2*t2*np.sin(phi)*np.sin(2*p)
    d2=-2*t1*np.cos(p)
    d3=-t1+p*0
    d4=2*t2*np.cos(p+phi)
    d5=2*t2*np.cos(p-phi)
    d6=2*t2*np.cos(phi)*np.cos(2*p)
    a=np.zeros((Np+1,2*N,2*N))
    b=np.zeros(Np+1)
    for i in range(N):
        a[:,i,i]=d1[:]+d6[:]
        a[:,i+N,i+N]=-d1[:]+d6[:]
        a[:,i+N,i]=d2[:]
        a[:,i,i+N]=d2[:]
    for i in range(N-1):
        a[:,i+1,i]=d4[:]
        a[:,i,i+1]=d4[:]
        a[:,i+N+1,i+N]=d5[:]
        a[:,i+N,i+N+1]=d5[:]
        a[:,i+1,i+N]=d3[:]
        a[:,i+N,i+1]=d3[:]
    b,c=np.linalg.eig(a)
    d=np.linalg.inv(c)       
    M=np.square(d)
    t=M[:,:,e]
    b=t[:,0]
    return b    