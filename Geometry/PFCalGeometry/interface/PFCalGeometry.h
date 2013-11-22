#ifndef PFCalGeometry_PFCalGeometry_h
#define PFCalGeometry_PFCalGeometry_h

/**
   \class PFCalGeometry
   PFCal geometry model
   \author P. Silva - CERN
*/

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <vector>
#include <map>

class GeomDetType;
class GeomDetUnit;

class PFCalGeometry {

 public:

  typedef std::vector<GeomDetType*>          DetTypeContainer;
  typedef std::vector<GeomDet*>              DetContainer;
  typedef std::vector<GeomDetUnit*>          DetUnitContainer;
  typedef std::vector<DetId>                 DetIdContainer;
  typedef  __gnu_cxx::hash_map< unsigned int, GeomDet*>     mapIdToDet;

  /// Default constructor
  PFCalGeometry();

  /// Destructor
  virtual ~PFCalGeometry();

  // Return a vector of all det types
  virtual const DetTypeContainer& detTypes() const;
  
  // Return a vector of all GeomDetUnit
  virtual const DetUnitContainer& detUnits() const;

  // Return a vector of all GeomDet
  virtual const DetContainer& dets() const;
  
  // Return a vector of all GeomDetUnit DetIds
  virtual const DetIdContainer& detUnitIds() const;

  // Return a vector of all GeomDet DetIds
  virtual const DetIdContainer& detIds() const;

  // Return the pointer to the GeomDetUnit corresponding to a given DetId
  virtual const GeomDetUnit* idToDetUnit(DetId) const;

  // Return the pointer to the GeomDet corresponding to a given DetId
  virtual const GeomDet* idToDet(DetId) const;


 private:

  DetContainer m_dets;
  DetUnitContainer m_partitions;
  DetTypeContainer m_partitionTypes;
  DetIdContainer m_partitionIds,  m_detIds;

  mapIdToDet m_detIdDict;
};

#endif
