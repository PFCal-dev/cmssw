import ROOT
import os
import optparse
import re
from sensorEquivalentMap import *

PLOTTITLES={'maxcounts' : 'Hottest wafer occupancy',
            'maxcounts0': '1^{st} hottest wafer neighbor',
            'maxcounts1': '2^{nd} hottest wafer neighbor',
            'maxcounts2': '3^{rd} hottest wafer neighbor',
            'maxcounts3': '4^{th} hottest wafer neighbor',
            'maxcounts4': '5^{th} hottest wafer neighbor',
            'maxcounts5': '6^{th} hottest wafer neighbor',
            'adc'       : 'Energy (MIP eq.)',
            'counts'    : 'Occupancy'
        }


def getPlotsIn(url,title,pfix):

    waferPlots={}
    inF=ROOT.TFile.Open(url)
    d=inF.Get('ana')

    #loop over directories
    for k in d.GetListOfKeys():
        sd=k.ReadObj()
        sdname=k.GetName()

        #parse sub-det, layer, u, v from name
        vals = [int(d) for d in re.findall(r'-?\d+', sdname)]
        waferKey=tuple(vals[0:4])

        waferPlots[waferKey]={}

        #save histos
        for kk in sd.GetListOfKeys():
            hname=kk.GetName()
            pname=hname.replace(sdname+'_','')
            waferPlots[waferKey][pname]=kk.ReadObj().Clone(hname+pfix)
            waferPlots[waferKey][pname].SetDirectory(0)
            waferPlots[waferKey][pname].SetTitle(title)
            waferPlots[waferKey][pname].SetLineWidth(2)
            
    return waferPlots

def getQuantiles(plotColl,q=0.5):

    momentSummary=[]
    for i in range(len(plotColl)):
        momentSummary.append({})
        for p in plotColl[i]:
            plotColl[i][p].SetBinContent(1,0) #FIXME in the analyzer
            qval=np.array([0.])
            if plotColl[i][p].Integral()>0:
                prob=np.array([q])
                plotColl[i][p].GetQuantiles(1,qval,prob)
            momentSummary[i][p]=qval[0]

    return momentSummary

def showSummaryForWafer(plotColl,extraText,pname,nPerRow=3):

    nPlots=len(plotColl[0])
    pnames=plotColl[0].keys()
    nx=nPerRow
    ny=nPlots/nPerRow
    if nx*ny<nPlots : ny+=1
    dx=1./float(nPerRow)
    dy=1./float(ny)

    c=ROOT.TCanvas('c','c',1500,ny*300)
    c.SetTopMargin(0)
    c.SetLeftMargin(0)
    c.SetRightMargin(0)
    c.SetBottomMargin(0)

    pads=[]
    colors=[ROOT.kBlack, ROOT.kMagenta, ROOT.kMagenta+2, ROOT.kMagenta-9,ROOT.kRed+1,ROOT.kAzure+7, ROOT.kBlue-7]
    for ix in range(0,nx):
        for iy in range(0,ny):
            c.cd()
            idx=len(pads)
            plot=pnames[idx]
            plotTitle=PLOTTITLES[plot]
            pads.append( ROOT.TPad(plot,plot,ix*dx,1-iy*dy,(ix+1)*dx,1-(iy+1)*dy) )            
            pads[-1].SetTopMargin(0.06)
            pads[-1].SetLeftMargin(0.12)
            pads[-1].SetRightMargin(0.02)
            pads[-1].SetBottomMargin(0.12)
            pads[-1].Draw()
            pads[-1].cd()
            pads[-1].SetLogy()

            drawOpt='hist'        
            for i in range(len(plotColl)):
                plotColl[i][plot].Draw(drawOpt)
                plotColl[i][plot].SetLineColor(colors[i])
                plotColl[i][plot].GetXaxis().SetLabelSize(0.05)
                plotColl[i][plot].GetXaxis().SetTitleSize(0.05)
                plotColl[i][plot].GetYaxis().SetLabelSize(0.05)
                plotColl[i][plot].GetYaxis().SetTitleSize(0.05)
                drawOpt='histsame'

            if ix==0 and iy==0:
                leg=pads[-1].BuildLegend(0.6,0.9,0.9,0.8)
                leg.SetBorderSize(0)
                leg.SetTextFont(42)
                leg.SetTextSize(0.05)
                leg.SetFillStyle(0)
                
                tex=ROOT.TLatex()
                tex.SetTextFont(42)
                tex.SetTextSize(0.05)
                tex.SetNDC()
                tex.DrawLatex(0.12,0.96,'#bf{CMS} #it{simulation preliminary}')
                tex.SetTextSize(0.04)
                for i in range(len(extraText)): 
                    tex.DrawLatex(0.15,0.9-i*0.05,extraText[i])

            titletex=ROOT.TLatex()
            titletex.SetTextFont(62)
            titletex.SetTextSize(0.05)
            titletex.SetNDC()
            titletex.SetTextAlign(ROOT.kHAlignRight+ROOT.kVAlignCenter)
            titletex.DrawLatex(0.95,0.98,plotTitle)

    c.Modified()
    c.Update()
    c.SaveAs(pname+'.png')
    for p in pads: p.Delete()

def showMomentSummary(momentSummary,sensorPos,outdir,idx=0):

    #check the available distributions
    dists=momentSummary[momentSummary.keys()[0]][idx].keys()
    
    import csv
    fOut=open(os.path.join(outdir,'occupancy_summary.dat'),'w')
    csv_writer = csv.writer(fOut, delimiter=',')

    #loop over each sub-detector layer
    subdets=set([x[0] for x in sensorPos])
    for sd in subdets:
        layers=set( [x[1] for x in sensorPos if x[0]==sd] )
        for lay in layers:

            for d in dists:
                
                if not 'counts' in d : continue

                uvzlist=[]
                labels=[]
                for waferKey in momentSummary:
                    isd,ilay,iu,iv=waferKey
                    if isd!=sd or ilay!=lay :continue

                    ncells,r,z,eta,phi=sensorPos[waferKey]               
                    occ=float(momentSummary[waferKey][idx][d])/float(ncells)

                    uvzlist.append( [iu,iv,occ] )
                    labels.append( '%d'%(100*occ) )

                if len(uvzlist)==0: continue
                extraText=[ PLOTTITLES[d],
                            '%s layer %d'%('CEE' if sd==0 else 'CEH', lay)
                            ]
                drawSensorEquivalentMap(uvzlist=uvzlist,
                                        labels=labels,
                                        outname=os.path.join(outdir,'%s_sd%d_lay%d'%(d,sd,lay)),
                                        extraText=extraText,
                                        cmapName='Wistia',
                                        zran=[0,1])
                
                #write
                if d=='counts':
                    for waferData in uvzlist:
			ncells=sensorPos[(sd,lay,waferData[0],waferData[1])][0]
                        csv_writer.writerow( [sd,lay,1 if ncells>400 else 0]+waferData )

    fOut.close()



def main():

    #parse inputs
    #configuration
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-o', '--out',   dest='output',  help='output directory [%default]',  default='plots', type='string')    
    parser.add_option('--noWaferPlots',   dest='noWaferPlots',  help='disable wafer plots [%default]',  default=False, action='store_true')
    (opt, args) = parser.parse_args()

    #define inputs
    procList=[ x.split('=') for x in args ]    
    plots=[]
    for i in range(len(args)):
        title,url=procList[i]
        plots.append( getPlotsIn(url=url,title=title,pfix='_%d'%i) )

    #get the sensor position map
    cmssw=os.environ['CMSSW_BASE']
    sensorPos=parseWaferPositionsMap(url='%s/src/SimCalorimetry/HGCalSimAlgos/test/scripts/wafer_pos.dat'%cmssw)

    #prepare summary
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gROOT.SetBatch(True)
    os.system('mkdir -p '+opt.output)

    momentSummary={}
    for waferKey in plots[0]:
        if not waferKey in sensorPos : continue

        ncells,r,z,eta,phi=sensorPos[waferKey]
        if abs(eta)<1.5 or abs(eta)>3 : continue

        plotColl=[ x[waferKey] for x in plots]

        momentSummary[waferKey]=getQuantiles(plotColl=plotColl,q=0.5)
        
        if opt.noWaferPlots: continue
        pname='summary_sd%d_lay%d_%d_%d'%waferKey
        pname=pname.replace('-','m')
        extraText=[
            '%s layer %d'%('CEE' if waferKey[0]==0 else 'CEH', waferKey[1]),
            '(u,v)=(%s,%s)'%(waferKey[2],waferKey[3]),
            'R=%3.2f z=%3.2f'%(r,z),
            '#eta=%3.2f #phi=%3.2f'%(eta,phi)
        ]
        showSummaryForWafer(plotColl=plotColl,
                            extraText=extraText,
                            pname=os.path.join(opt.output,pname)
                            )

    showMomentSummary(momentSummary,sensorPos,opt.output)




if __name__ == "__main__":
    main()
