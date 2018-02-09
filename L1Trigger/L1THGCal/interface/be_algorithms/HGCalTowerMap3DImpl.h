#ifndef __L1Trigger_L1THGCal_HGCalTowerMap3DImpl_h__
#define __L1Trigger_L1THGCal_HGCalTowerMap3DImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"


class HGCalTowerMap3DImpl{

 public:

  HGCalTowerMap3DImpl( const edm::ParameterSet &conf); 
  
  void buildTowerMapMulticlusters( const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtr,
				   l1t::HGCalMulticlusterBxCollection & towerMaps
		       );
  
 private:

  static constexpr double kEtaMin_ = 1.479;
  static constexpr double kEtaMax_ = 3.;
  static constexpr double kPhiMin_ = -M_PI;
  static constexpr double kPhiMax_ = +M_PI;

  unsigned nEtaBins_;
  unsigned nPhiBins_;
  vector<double> etaBins_;
  vector<double> phiBins_;


};



#endif
