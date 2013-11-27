/**
   @class PFCalCell
   @short wrapper for a sensitive volume. For now it doesn't do much ...
   @author P. Silva - CERN
 */

#ifndef _pfcalcell_h_
#define _pfcalcel_h_

#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/PFCalGeometry/interface/PFCalCellSpecs.h"

class PFCalCell : public GeomDetUnit
{
public:
  
  /**
     @short CTOR
   */
  PFCalCell(DetId id, BoundPlane::BoundPlanePointer bp, PFCalCellSpecs *specs);

  /**
     @short
   */
  const Topology& topology() const;

  /**
     @short
   */
  const GeomDetType& type() const; 

  /**
     @short DTOR
   */
  ~PFCalCell();

  /**
     @short DetId getter
   */
  DetId id() const { return id_; }

private:

  DetId id_;
  PFCalCellSpecs *top_;

};


#endif
