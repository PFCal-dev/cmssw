#ifndef _PFCalGeometryAnalyzer_h_
#define _PFCalGeometryAnalyzer_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include <string>

/**
   @class PFCalGeometryAnalyzer
   @short test unit to scan PFCal geometry
   @author P. Silva (CERN)
*/

class PFCalGeometryAnalyzer : public edm::EDAnalyzer 
{
  
 public:
  
  explicit PFCalGeometryAnalyzer( const edm::ParameterSet& );
  ~PFCalGeometryAnalyzer();
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

 private:
  
  void listRoots(const DDCompactView  &);
  void showSerialized(const DDCompactView  &dd, const std::string & root);

  std::string ddViewName_,ddRootNodeName_;
  bool listRoots_,showSerialized_,runGeometryInfoDump_,runGeometryBuilderFromDDD_;
  edm::ParameterSet geomInfoDumpCfg_;

};
 

#endif
