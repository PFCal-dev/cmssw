/** \file Implementation of the PFCalGeometryProducer
 *
 * \author P. Silva - CERN
 */

#include "Geometry/PFCalGeometryBuilder/plugins/PFCalGeometryESModule.h"
#include "Geometry/PFCalGeometryBuilder/interface/PFCalGeometryBuilderFromDDD.h"

#include <Geometry/Records/interface/IdealGeometryRecord.h>

#include <DetectorDescription/Core/interface/DDCompactView.h>

#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>

#include <memory>

using namespace edm;

//
PFCalGeometryESModule::PFCalGeometryESModule(const edm::ParameterSet & p){
  setWhatProduced(this);
}


//
PFCalGeometryESModule::~PFCalGeometryESModule(){}

//
boost::shared_ptr<PFCalGeometry> PFCalGeometryESModule::produce(const PFCalGeometryRecord & record){

  //instantiate a geometry builder from DDD and put the result to the event setup
  PFCalGeometryBuilderFromDDD builder;
  edm::ESTransientHandle<DDCompactView> cpv;
  //record.getRecord<IdealGeometryRecord>().get(cpv);
  
  return boost::shared_ptr<PFCalGeometry>(builder.build(&(*cpv)));
}

DEFINE_FWK_EVENTSETUP_MODULE(PFCalGeometryESModule);
