import ROOT


def float_equal(x1, x2):
    prec = 1.e-4
    if abs(x1)<prec and abs(x2)>prec: return False
    elif abs(x1)<prec and abs(x2)<prec: return True
    else: return abs( (x1-x2)/x1)<prec

def compare_lines(line1, line2):
    xy11 = (line1.GetX1(), line1.GetY1())
    xy12 = (line1.GetX2(), line1.GetY2())
    xy21 = (line2.GetX1(), line2.GetY1())
    xy22 = (line2.GetX2(), line2.GetY2())
    samecorner1 = (float_equal(xy11[0],xy21[0]) and float_equal(xy11[1],xy21[1])) or (float_equal(xy11[0],xy22[0]) and float_equal(xy11[1],xy22[1]))
    samecorner2 = (float_equal(xy12[0],xy21[0]) and float_equal(xy12[1],xy21[1])) or (float_equal(xy12[0],xy22[0]) and float_equal(xy12[1],xy22[1]))
    #if prt: print "[",xy11,xy12,"]","[",xy21,xy22,"]",(samecorner1 and samecorner2)
    return samecorner1 and samecorner2

def boxlines(box):
    lines = []
    lines.append(ROOT.TLine(box.GetX1(), box.GetY1(), box.GetX1(), box.GetY2()))
    lines.append(ROOT.TLine(box.GetX1(), box.GetY1(), box.GetX2(), box.GetY1()))
    lines.append(ROOT.TLine(box.GetX1(), box.GetY2(), box.GetX2(), box.GetY2()))
    lines.append(ROOT.TLine(box.GetX2(), box.GetY1(), box.GetX2(), box.GetY2()))
    return lines

class Position:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.

class Cell:
    def __init__(self):
        self.id = 0
        self.zside = 0
        self.layer = 0
        self.sector = 0
        self.center = Position()
        self.corners = [Position(), Position(), Position(), Position()]

    def box(self):
        return ROOT.TBox(self.corners[0].x, self.corners[0].y, self.corners[2].x, self.corners[2].y)

    def __eq__(self, other):
        return self.id==other.id

    def __lt__(self, other):
        return self.id<other.id

    def __le__(self, other):
        return self.id<=other.id

class TriggerCell:
    def __init__(self):
        self.id = 0
        self.zside = 0
        self.layer = 0
        self.sector = 0
        self.module = 0
        self.triggercell = 0
        self.center = Position()
        self.cells = []
        self.borderlines = []

    def fillLines(self):
        for cell in self.cells:
            box = cell.box()
            thisboxlines = boxlines(box)
            for boxline in thisboxlines:
                existingline = None
                for line in self.borderlines:
                    if compare_lines(boxline, line):
                        existingline = line
                        break
                if existingline:
                    self.borderlines.remove(existingline)
                else:
                    self.borderlines.append(boxline)

class Module:
    def __init__(self):
        self.id = 0
        self.zside = 0
        self.layer = 0
        self.sector = 0
        self.module = 0
        self.center = Position()
        self.cells = []
        self.borderlines = []

    def fillLines(self):
        for cell in self.cells:
            for cellline in cell.borderlines:
                existingline = None
                for line in self.borderlines:
                    if compare_lines(cellline, line):
                        existingline = line
                        break
                if existingline:
                    self.borderlines.remove(existingline)
                else:
                    self.borderlines.append(cellline)



