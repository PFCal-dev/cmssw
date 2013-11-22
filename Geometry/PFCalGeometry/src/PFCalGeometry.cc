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
  return m_partitionTypes;
}

//
const PFCalGeometry::DetUnitContainer& PFCalGeometry::detUnits() const{
  return m_partitions;
}

//
const PFCalGeometry::DetContainer& PFCalGeometry::dets() const{
  return m_dets;
}

//
const PFCalGeometry::DetIdContainer& PFCalGeometry::detUnitIds() const{
  return m_partitionIds;
}

//
const PFCalGeometry::DetIdContainer& PFCalGeometry::detIds() const {
  return m_detIds;
}

//
const GeomDetUnit* PFCalGeometry::idToDetUnit(DetId id) const
{
  return dynamic_cast<const GeomDetUnit*>(idToDet(id));
}

//
const GeomDet* PFCalGeometry::idToDet(DetId id) const
{
  mapIdToDet::const_iterator i = m_detIdDict.find(id);
  return (i != m_detIdDict.end()) ? i->second : 0 ;
}
