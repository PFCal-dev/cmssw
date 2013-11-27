/**
   @short implementation of PFCalCell
   @author P. Silva - CERN
 */

#include "Geometry/PFCalGeometry/interface/PFCalCell.h"

//
PFCalCell::PFCalCell(DetId id, BoundPlane::BoundPlanePointer bp, PFCalCellSpecs *specs) :
  GeomDetUnit(bp), id_(id), top_(specs)
{
}

//
PFCalCell::~PFCalCell()
{
  delete top_;
}

//
const Topology& PFCalCell::topology() const
{
  return top_->topology();
}

//
const GeomDetType &PFCalCell::type() const
{
  return (*top_);
}


