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
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "Geometry/PFCalGeometry/interface/PFCalCell.h"

#include <vector>
#include <map>

class GeomDetType;
class GeomDetUnit;

class PFCalGeometry : public TrackingGeometry {

 public:

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

  void add(PFCalCell *);

  void print();

 private:

  DetContainer cells_;
  DetIdContainer cellDetIds_;
  DetTypeContainer cellTypes_;
  DetUnitContainer cellUnits_;


  mapIdToDet detIdDict_;

};

#endif
