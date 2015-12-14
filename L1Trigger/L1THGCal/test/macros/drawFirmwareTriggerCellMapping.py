import ROOT
import random
from DrawingUtilities import *

### PARAMETERS #####
layer  = 15
sector = 1
zside  = 1
mod    = 10
inputFileName = "../test.root"
firmwareMappingFile = "../cell_map_layer15_module10.txt"
outputName    = "firmware_triggerCell_map"
####################



inputFile = ROOT.TFile.Open(inputFileName)
treeModules      = inputFile.Get("hgcaltriggergeomtester/TreeModules")
treeTriggerCells = inputFile.Get("hgcaltriggergeomtester/TreeTriggerCells")
treeCells        = inputFile.Get("hgcaltriggergeomtester/TreeCells")
treeModules.__class__ = ROOT.TTree
treeTriggerCells.__class__ = ROOT.TTree
treeCells.__class__ = ROOT.TTree

## filling cell map
cells = {}
cut = "layer=={0} && sector=={1} && zside=={2}".format(layer,sector,zside)
treeCells.Draw(">>elist1", cut, "entrylist")
entryList1 = ROOT.gDirectory.Get("elist1")
entryList1.__class__ = ROOT.TEntryList
nentry = entryList1.GetN()
treeCells.SetEntryList(entryList1)
for ie in xrange(nentry):
    if ie%10000==0: print "Entry {0}/{1}".format(ie, nentry)
    entry = entryList1.GetEntry(ie)
    treeCells.GetEntry(entry)
    cell = Cell()
    cell.id       = treeCells.id
    cell.zside    = treeCells.zside
    cell.layer    = treeCells.layer
    cell.sector   = treeCells.sector
    cell.center.x = treeCells.x
    cell.center.y = treeCells.y
    cell.center.z = treeCells.z
    cell.corners[0].x = treeCells.x1
    cell.corners[0].y = treeCells.y1
    cell.corners[1].x = treeCells.x2
    cell.corners[1].y = treeCells.y2
    cell.corners[2].x = treeCells.x3
    cell.corners[2].y = treeCells.y3
    cell.corners[3].x = treeCells.x4
    cell.corners[3].y = treeCells.y4
    if cell.id not in cells: cells[cell.id] = cell

## filling trigger cell map
triggercells = {}
treeTriggerCells.Draw(">>elist2", cut, "entrylist")
entryList2 = ROOT.gDirectory.Get("elist2")
entryList2.__class__ = ROOT.TEntryList
nentry = entryList2.GetN()
treeTriggerCells.SetEntryList(entryList2)
for ie in xrange(nentry):
    if ie%10000==0: print "Entry {0}/{1}".format(ie, nentry)
    entry = entryList2.GetEntry(ie)
    treeTriggerCells.GetEntry(entry)
    triggercell = TriggerCell()
    triggercell.id       = treeTriggerCells.id
    triggercell.zside    = treeTriggerCells.zside
    triggercell.layer    = treeTriggerCells.layer
    triggercell.sector   = treeTriggerCells.sector
    triggercell.module   = treeTriggerCells.module
    triggercell.triggercell = treeTriggerCells.triggercell
    triggercell.center.x = treeTriggerCells.x
    triggercell.center.y = treeTriggerCells.y
    triggercell.center.z = treeTriggerCells.z
    for cellid in treeTriggerCells.c_id:
        if not cellid in cells: raise StandardError("Cannot find cell {0} in trigger cell".format(cellid))
        cell = cells[cellid]
        triggercell.cells.append(cell)
    triggercells[triggercell.id] = triggercell

for id,triggercell in triggercells.items():
    triggercell.fillLines()

## filling module map
modules = {}
treeModules.Draw(">>elist3", cut, "entrylist")
entryList3 = ROOT.gDirectory.Get("elist3")
entryList3.__class__ = ROOT.TEntryList
nentry = entryList3.GetN()
treeModules.SetEntryList(entryList3)
for ie in xrange(nentry):
    if ie%10000==0: print "Entry {0}/{1}".format(ie, nentry)
    entry = entryList3.GetEntry(ie)
    treeModules.GetEntry(entry)
    module = Module()
    module.id       = treeModules.id
    module.zside    = treeModules.zside
    module.layer    = treeModules.layer
    module.sector   = treeModules.sector
    module.module   = treeModules.module
    module.center.x = treeModules.x
    module.center.y = treeModules.y
    module.center.z = treeModules.z
    for cellid in treeModules.tc_id:
        if not cellid in triggercells: raise StandardError("Cannot find trigger cell {0} in module".format(cellid))
        cell = triggercells[cellid]
        module.cells.append(cell)
    modules[module.id] = module

for id,module in modules.items():
    module.fillLines()

print "Read", len(cells), "cells" 
print "Read", len(triggercells), "trigger cells"
print "Read", len(modules), "modules"

# Read firmware mapping file
firmwareMapping = []
with open(firmwareMappingFile) as f:
    for line in f:
        firmwareMapping.append(int(line))
firmwareMapping.reverse()

# Retrieve cells contained in the chosen module and sort them
cellsInModule = []
for id,module in modules.items():
    if module.module==mod:
        for triggercell in module.cells:
            for cell in triggercell.cells:
                cellsInModule.append(cell)
cellsInModule.sort()

# Create trigger cells and module from information in the firmware mapping file
triggercellsforfirmware = []
moduleforfirmware = Module()
for i,index in enumerate(firmwareMapping):
    if i%4==0: triggercellsforfirmware.append(TriggerCell()) # FIXME: works only for trigger cells containing 4 cells
    cell = cellsInModule[index]
    triggercellsforfirmware[-1].cells.append(cell)

for triggercell in triggercellsforfirmware:
    moduleforfirmware.cells.append(triggercell)
    triggercell.fillLines()
moduleforfirmware.fillLines()

## create output canvas
outputFile = ROOT.TFile.Open(outputName+".root", "RECREATE")
maxx = -99999.
minx = 99999.
maxy = -99999.
miny = 99999.
for id,triggercell in triggercells.items():
    x = triggercell.center.x
    y = triggercell.center.y
    if x>maxx: maxx=x
    if x<minx: minx=x
    if y>maxy: maxy=y
    if y<miny: miny=y
minx = minx*0.8 if minx>0 else minx*1.2
miny = miny*0.8 if miny>0 else miny*1.2
maxx = maxx*1.1 if maxx>0 else maxx*0.9
maxy = maxy*1.2 if maxy>0 else maxy*0.8
canvas = ROOT.TCanvas("triggerCellMap", "triggerCellMap", 1400, int(1400*(maxy-miny)/(maxx-minx)))
canvas.Range(minx, miny, maxx, maxy)

## Print cells
drawstyle = "lf"
boxes = []
for id,module in modules.items():
    modulelines = []
    for triggercell in module.cells:
        for cell in triggercell.cells:
            box = cell.box()
            box.SetFillColor(0)
            box.SetLineColor(ROOT.kGray)
            box.Draw(drawstyle)
            boxes.append(box)
            if not "same" in drawstyle: drawstyle += " same"

# Print trigger cells and module that have been built from firmware mapping file
linesTrigger = []
for triggercell in triggercellsforfirmware:
    linesTrigger.extend(triggercell.borderlines)

for line in linesTrigger:
    line.Draw()
for line in moduleforfirmware.borderlines:
    line.SetLineColor(ROOT.kBlue)
    line.SetLineWidth(3)
    line.Draw()



canvas.Write()
canvas.Print(outputName+".png")


inputFile.Close()




