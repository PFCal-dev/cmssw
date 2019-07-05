import ROOT
import sys
import random

def compareDigis(t):
    
    """loops over events, matches the two collections digi by digi, does some plotting"""
    
    nEvents=t.GetEntriesFast()
    plots={'matched'      : ROOT.TGraph(),
           'cpu'          : ROOT.TGraph(),
           'gpu'          : ROOT.TGraph(),        
           'cpu_digis'    : ROOT.TH2F('cpudigis', ';Event number;Energy [ADC];Channels',nEvents,0,nEvents,25,0,25),
           'gpu_digis'    : ROOT.TH2F('gpudigis', ';Event number;Energy [ADC];Channels',nEvents,0,nEvents,25,0,25),
           'cpu_toa'      : ROOT.TH2F('cpu_toa',  ';Event number;Toa;Channels',nEvents,0,nEvents,100,0,100),
           'gpu_toa'      : ROOT.TH2F('gpu_toa',  ';Event number;Toa;Channels',nEvents,0,nEvents,100,0,100),
           'digi_diff'    : ROOT.TH2F('digi_diff',';Event number;#DeltaEnergy [ADC];Channels',nEvents,0,nEvents,40,-20.5,19.5),
           'digi_diff_vsen' : ROOT.TH2F('digi_diff_vsen',';Energy[ADC] ;#DeltaEnergy [ADC];Channels',100,0,100,40,-20.5,19.5),
           'toa_diff'       : ROOT.TH2F('toa_diff',';Event number;#DeltaToa;Channels',nEvents,0,nEvents,20,-10,10),
           'gpu_digis_tail' : ROOT.TH2F('gpu_digis_tail',';Layer;Energy [ADC];Channels',52,0,52,100,0,100),
           }
    plots['matched'].SetName('matched')
    plots['cpu'].SetName('cpu')
    plots['gpu'].SetName('gpu')

    for n in range(nEvents):
        t.GetEntry(n)
        print n,'/',nEvents

        digis={}

        #cpu digis
        for i in range(t.HGCDigiIndex.size()):
            idx   = t.HGCDigiIndex[i]           
            if idx>1 : continue
            pos   = True if t.HGCDigiPosz[i]>0 else False
            layer = t.HGCDigiLayer[i]
            wu    = t.HGCDigiWaferU[i]
            wv    = t.HGCDigiWaferV[i]
            u     = t.HGCDigiCellU[i]
            v     = t.HGCDigiCellV[i]      
            key   = (pos,idx,layer,wu,wv,u,v)      
            digis[key] = [t.HGCDigiToa[i],t.HGCDigiSamples[i][2],None,None]

        #gpu digis
        nOnlyGPU=0
        for i in range(t.GPUHGCDigiIndex.size()):
            idx   = t.GPUHGCDigiIndex[i]
            if idx>1 : continue
            pos   = True if t.GPUHGCDigiPosz[i]>0 else False
            layer = t.GPUHGCDigiLayer[i]            
            wu    = t.GPUHGCDigiWaferU[i]
            wv    = t.GPUHGCDigiWaferV[i]
            u     = t.GPUHGCDigiCellU[i]
            v     = t.GPUHGCDigiCellV[i]            
            key   = (pos,idx,layer,wu,wv,u,v)
            if key in digis:
                digis[key][2] = t.GPUHGCDigiToa[i]
                digis[key][3] = t.GPUHGCDigiSamples[i][2]            
                plots['toa_diff'].Fill(n,digis[key][2]-digis[key][0])
                plots['digi_diff'].Fill(n,digis[key][3]-digis[key][1])

                en=random.choice( [digis[key][1],digis[key][3]] )
                rel_den=digis[key][3]-digis[key][1]
                plots['digi_diff_vsen'].Fill(en,rel_den)

                #if en<10 and abs(rel_den)>40:
                #    plots['gpu_digis_tail'].Fill(layer+28*idx,digis[key][3])
                #    print n,en,digis[key][3],key

            else:
                digis[key]=[None,None,t.GPUHGCDigiToa[i],t.GPUHGCDigiSamples[i][2]]
                plots['gpu_toa'].Fill(n,digis[key][2])
                plots['gpu_digis'].Fill(n,digis[key][3])
                nOnlyGPU+=1

        nMatch=sum([1 if not None in digis[x] else 0 for x in digis])

        nOnlyCPU=0
        for x in digis:
            if None in digis[x][0:2] : continue
            if None in digis[x][2:] :
                nOnlyCPU+=1            
                plots['cpu_toa'].Fill(n,digis[x][0])
                plots['cpu_digis'].Fill(n,digis[x][1])

        plots['matched'].SetPoint(n,n,nMatch)
        plots['cpu'].SetPoint(n,n,nOnlyCPU)
        plots['gpu'].SetPoint(n,n,nOnlyGPU)

    return plots

def showPlots(summaryFile,header):

    def cmsHeader(c,doLeg=False,extraTxt=[]):       
        if doLeg:
            leg=c.BuildLegend()
            leg.SetFillStyle(0)
            leg.SetTextFont(42)
            leg.SetTextSize(0.035)
            leg.SetBorderSize(0)
        tex=ROOT.TLatex()
        tex.SetTextFont(42)
        tex.SetTextSize(0.04)
        tex.SetNDC()
        tex.DrawLatex(0.12,0.96,'#bf{CMS} #it{simulation preliminary} @ 6^{th} Patatrack')        
        tex.SetTextAlign(ROOT.kHAlignRight+ROOT.kVAlignCenter)
        if len(header)>0:
            tex.DrawLatex(0.96,0.975,header)

        tex.SetTextSize(0.035)
        for i in range(len(extraTxt)):
            tex.DrawLatex(0.84,0.9-0.06*i,extraTxt[i])
        c.Modified()
        c.Update()

    inF=ROOT.TFile.Open(summaryFile)
    c=ROOT.TCanvas('c','c',500,500)
    c.SetTopMargin(0.05)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.03)
    c.SetBottomMargin(0.1)

    #channel count
    grList=[]
    mg=ROOT.TMultiGraph()
    for k,ci,title in [('matched', 1,             'Matched'),
                       ('cpu',     2,             'CPU only'),
                       ('gpu',     ROOT.kGreen+1, 'GPU only')]:
        grList.append( inF.Get(k) )
        grList[-1].SetLineColor(ci)
        grList[-1].SetLineWidth(2)
        grList[-1].SetMarkerStyle(1)
        grList[-1].SetMarkerColor(ci)
        grList[-1].SetTitle(title)
        mg.Add(grList[-1],'c')
    mg.Draw('ac')
    mg.GetYaxis().SetTitle('Number of channels')
    mg.GetXaxis().SetTitle('Event number')
    mg.Draw('ac')
    cmsHeader(c,True)
    c.SaveAs('digicount.png')

    #resolution
    for prof,title in [('digi_diff','CPU vs GPU'),
                       ('digi_diff_vsen','CPU vs GPU'),
                       ('cpudigis','CPU only'),
                       ('gpudigis','GPU only'),
                       ('toa_diff','CPU vs GPU')]:
        
        h=inF.Get(prof)
        h.Draw('colz')
        c.SetRightMargin(0.15)
        h.GetZaxis().SetTitleOffset(-0.3)
        cmsHeader(c,False,[title])
        c.SaveAs(prof+'.png')

        hprof=h.ProjectionY()
        hprof.Draw()
        hprof.GetYaxis().SetTitle(h.GetZaxis().GetTitle())
        extraTxt=[title,
                  'Mean: %3.2f#pm%3.2f'%(hprof.GetMean(),hprof.GetMeanError()),
                  'RMS: %3.2f#pm%3.2f'%(hprof.GetRMS(),hprof.GetRMSError())]                  
        c.SetRightMargin(0.03)
        cmsHeader(c,False,extraTxt)
        c.SaveAs(prof+'_proj.png')

        
def main():

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gROOT.SetBatch(True)

    inFile=sys.argv[1]
    header=sys.argv[2] if len(sys.argv)>2 else ''

    #build summary file
    summaryFile='hgcpatadigis_summary.root'
    if not 'hgcpatadigis_summary' in inFile:
        fIn=ROOT.TFile.Open(sys.argv[1])
        t=fIn.Get('hgcalTupleTree/tree')
        plots=compareDigis(t)
        
        #dump the plots
        fOut=ROOT.TFile.Open(summaryFile,'RECREATE')
        for x in plots:
            if plots[x].InheritsFrom('TH1'):
                plots[x].SetDirectory(fOut)
            plots[x].Write()
        fOut.Close()

        fIn.Close()
    else:
        summaryFile=inFile

    showPlots(summaryFile,header)


if __name__ == "__main__":
    # execute only if run as a script
    main()
