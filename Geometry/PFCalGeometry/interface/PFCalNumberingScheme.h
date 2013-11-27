/**
   \class PFCalNumberingScheme
   Defines the way to parse the geometry and assign a detId to a volume in PFCal
   \author P. Silva - CERN
 */

#ifndef _pfcalnumbering_scheme_h_
#define _pfcalnumbering_scheme_h_

#include "DataFormats/DetId/interface/DetId.h"

class PFCalNumberingScheme
{
 public :
  PFCalNumberingScheme () { }
  ~PFCalNumberingScheme() { }
  DetId buildDetIdFor(bool isPos,bool isHE, unsigned int layerId, unsigned int sectorId);

 private:

};



#endif
