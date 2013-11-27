#ifndef _pfcalcellspecs_h_
#define _pfcalcellspecs_h_

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <vector>
#include <string>

class PFCalCellSpecs : public GeomDetType
{
public:

  /**
  @short CTOR 
  */
  PFCalCellSpecs(const std::string& name,SubDetector sd,const std::vector<double> &solidPars);

  /**
     @short DTOR
   */
  ~PFCalCellSpecs();

  const Topology& topology() const { return *top_; }
  const std::string& detName() const { return name_; }

private:
  
  std::string name_;
  StripTopology* top_;
};

#endif
