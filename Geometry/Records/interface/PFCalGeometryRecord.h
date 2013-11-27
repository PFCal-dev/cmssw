#ifndef Records_PFCalGeometryRecord_h
#define Records_PFCalGeometryRecord_h

/** \class PFCalGeometryRecord
 *  The PFCal DetUnit geometry.
 *  \author P.Silva - CERN
 */


#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PFCalHERecoGeometryRcd.h"
#include "Geometry/Records/interface/PFCalEERecoGeometryRcd.h"
#include "boost/mpl/vector.hpp"


class PFCalGeometryRecord : public edm::eventsetup::DependentRecordImplementation<PFCalGeometryRecord,boost::mpl::vector<IdealGeometryRecord, PFCalHERecoGeometryRcd, PFCalEERecoGeometryRcd> > {};

#endif

