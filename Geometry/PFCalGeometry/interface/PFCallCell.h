/**
   @class PFCallCell
   @short wrapper for a sensitive volume
   @author P. Silva - CERN
 */

#ifndef _pfcalcell_h_
#define _pfcalcel_h_

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"


class PFCalCell : public GeomDetUnit
{
public:
  
  /**
     @short CTOR
   */
  PFCalCell(PFCalDetId id, BoundPlane::BoundPlanePointer bp);

  /**
     @short DTOR
   */
  ~PFCalCell();

  /**
     @short DetId getter
   */
  DetId id() const { return id_; }

  const Topology& topology() const;
  const StripTopology& specificTopology() const;

  const GeomDetType& type() const;

  /**
     @short get center
   */
  LocalPoint getCenter(int strip) const;

private:

  DetId id_;
};
