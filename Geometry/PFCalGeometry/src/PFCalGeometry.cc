#include "Geometry/PFCalGeometry/interface/PFCalGeometry.h"

/** Implementation of the Model for PFCal Geometry
 *
 * \author P. Silva - CERN
 */

#include <Geometry/PFCalGeometry/interface/PFCalGeometry.h>
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>

//
PFCalGeometry::PFCalGeometry(){}

//
PFCalGeometry::~PFCalGeometry(){}

//
const PFCalGeometry::DetTypeContainer& PFCalGeometry::detTypes() const{
  return cellTypes_;
}

//
const PFCalGeometry::DetUnitContainer& PFCalGeometry::detUnits() const{
  return cellUnits_;
}

//
const PFCalGeometry::DetContainer& PFCalGeometry::dets() const{
  return cells_;
}

//
const PFCalGeometry::DetIdContainer& PFCalGeometry::detUnitIds() const{
  return cellDetIds_;
}

//
const PFCalGeometry::DetIdContainer& PFCalGeometry::detIds() const {
  return cellDetIds_;
}

//
const GeomDetUnit* PFCalGeometry::idToDetUnit(DetId id) const
{
  return dynamic_cast<const GeomDetUnit*>(idToDet(id));
}

//
const GeomDet* PFCalGeometry::idToDet(DetId id) const
{
  mapIdToDet::const_iterator i = detIdDict_.find(id);
  return (i != detIdDict_.end()) ? i->second : 0 ;
}

//
void PFCalGeometry::add(PFCalCell *cell)
{
  cells_.push_back(cell);
  cellDetIds_.push_back(cell->id());
  GeomDetType* t = const_cast<GeomDetType*>(&cell->type());
  cellTypes_.push_back( t );
  cellUnits_.push_back(cell);
  detIdDict_.insert(std::pair<DetId,GeomDetUnit*>(cell->geographicalId(),cell));
}


//
void PFCalGeometry::print()
{
  std::cout << "[PFCalGeometry] current geometry is" << std::endl;
  for(size_t i=0; i< cells_.size(); i++)
    {
      const PFCalCell *cell=dynamic_cast<const PFCalCell *>(cells_[i]);
      const PFCalCellSpecs &cellSpecs=dynamic_cast<const PFCalCellSpecs &>(cell->type());
      const StripTopology &top=dynamic_cast<const StripTopology &>(cellSpecs.topology());

      std::cout << cellSpecs.detName() << " 0x" << std::hex << cellDetIds_[i].rawId() << std::dec 
		<< " (" << top.localPosition(0).x() << "," << top.localPosition(0).y() << ","<< top.localPosition(0).z() << ")" << std::endl;
    }
  std::cout << std::endl;
}
