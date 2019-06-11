import ROOT
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon,Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

def getXY(u,v):
    return -2*u+v,np.sqrt(3.)*v

def getUV(x,y):

    v=int(round(y/np.sqrt(3.),0))
    u=int(round(-0.5*(x-v)))

    return u,v

def getSector(x,y):
    phi=np.mod(np.arctan2(y,x),2*np.pi)
    isector=0
    while True:
        if phi<np.pi/3.:
            break
        isector+=1
        phi-=np.pi/3.    
    return isector

def getEquivalentUV(x,y):    
    isector=getSector(x,y)
    phi=isector*np.pi/3.
    c,s=np.cos(-phi),np.sin(-phi)
    R=np.array(((c,-s),(s,c)))
    X=np.array((x,y))
    xyeq=R.dot(X)
    return getUV(xyeq[0],xyeq[1])

def parseWaferPositionsMap(url):

    sensorPos={}
    with open(url,'r') as f:
        for line in f:
            tokens=line.split()
            key=tuple([int(x) for x in tokens[0:4]])
            ncells=int(tokens[4])
            pos=[float(x) for x in tokens[5:]]
            sensorPos[key]=[ncells]+pos
    return sensorPos


def drawSensorEquivalentMap(uvzlist,labels,outname,extraText=[],cmapName='Pastel1',zran=None,labelSize=5):

    fig, ax = plt.subplots(1,figsize=(10, 10))
    ax.set_aspect('equal')

    #build the hexagonal mesh
    patches=[]
    cont_patches=[]
    xran=[0,0]
    yran=[0,0]
    radius=2./np.sqrt(3.)
    circleRadius=[]
    for i in range(len(uvzlist)):
        u,v,_=uvzlist[i]
        x,y=getXY(u,v)
        xran[0]=min(x-2*radius,xran[0])
        xran[1]=max(x+2*radius,xran[1])
        yran[0]=min(y-2*radius,yran[0])
        yran[1]=max(y+2*radius,yran[1])
        patches.append( RegularPolygon((x,y), 
                                       numVertices=6, 
                                       radius=radius,
                                       orientation=np.radians(60), 
                                       alpha=0.2, edgecolor='k') )
        
        if v==0: circleRadius.append( np.sqrt(x*x+y*y) )

        if len(labels)<i : continue
        label=labels[i]
        ax.text(x, y, label, ha='center', va='center', size=labelSize)

    patchColl=PatchCollection(patches)
    ax.add_collection(patchColl)
    
    #set the colours according to the z values
    norm_zvals=[uvz[2] for uvz in uvzlist]
    if zran:
        cnorm = Normalize(vmin=zran[0],vmax=zran[1])
    else:
        cnorm = Normalize(vmin=min(norm_zvals),vmax=max(norm_zvals))
    cmap = plt.get_cmap(cmapName)
    colors=cmap( cnorm(norm_zvals) )
    patchColl.set_color(colors)

    for r in circleRadius:
        plt.Circle((0, 0), r, color='gray', ls='--', fill=False)

    ax.autoscale_view()
    ax.set_xlabel('x',fontsize=16) 
    ax.set_ylabel('y',fontsize=16)        
    ax.set_xlim(xran[0],xran[1])
    ax.set_ylim(yran[0],yran[1])
    ax.text(0.05,1.02,'CMS preliminary', transform=ax.transAxes, fontsize=16)
    for i in range(len(extraText)):
        ax.text(0.05,0.95-i*0.05,extraText[i], transform=ax.transAxes, fontsize=14)

    fig.savefig(outname+'.png')


def dumpSensorEquivalentMap():

    fout=open('uvequiv.dat','w')

    uvzlist=[]
    labels=[]
    for u in range(-22,23):
        for v in range(-22,23):

            if u==0 and v==0: continue

            x,y=getXY(u,v)
            sector=getSector(x,y)
            ueq,veq=getEquivalentUV(x,y)
            fout.write('%d %d %d %d %d\n'%(u,v,sector,ueq,veq))

            radius=np.sqrt(x*x+y*y)
            if radius>30 : continue
        
            uvzlist.append( [u,v,1 if sector==0 else 0] )
            labels.append( '(%d,%d)'%(ueq,veq) )

            fout.write('%d %d %d %d %d\n'%(u,v,sector,ueq,veq))
    
    drawSensorEquivalentMap(uvzlist,labels,'waferequivalent')
    fout.close()


def main():
    dumpSensorEquivalentMap()

if __name__ == "__main__":
    main()
