#include "Geometry/PFCalGeometry/interface/PFCalCellSpecs.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

//
PFCalCellSpecs::PFCalCellSpecs(const std::string& name, SubDetector sd, const std::vector<double> &solidPars) :
  GeomDetType(name, sd),
  name_(name)
{
  double hhalf=solidPars[0]/cm;
  double bhalf=solidPars[4]/cm;
  double Bhalf=solidPars[8]/cm;
  float r0 = hhalf*(Bhalf + bhalf)/(Bhalf - bhalf);
  float striplength = hhalf*2;
  float pitch = (bhalf + Bhalf);
  int nstrip(1);
  top_ = new TrapezoidalStripTopology(nstrip, pitch, striplength, r0);
}

//
PFCalCellSpecs::~PFCalCellSpecs()
{

}
