#ifndef PFCalGeometry_PFCalGeometryESModule_h
#define PFCalGeometry_PFCalGeometryESModule_h

/** \class PFCalGeometryESModule
 *
 * ESProducer for PFCalGeometry 
 *
 * \author P. Silva - CERN
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/PFCalGeometry/interface/PFCalGeometry.h"
#include "Geometry/Records/interface/PFCalGeometryRecord.h"
#include "boost/shared_ptr.hpp"

class PFCalGeometryESModule : public edm::ESProducer {
 public:
  /// Constructor
  PFCalGeometryESModule(const edm::ParameterSet & p);

  /// Destructor
  virtual ~PFCalGeometryESModule();

  /// Produce PFCalGeometry.
  boost::shared_ptr<PFCalGeometry> produce(const PFCalGeometryRecord & record);

 private:
};
#endif
