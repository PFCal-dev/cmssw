#ifndef __L1Trigger_L1THGCal_HGCalTowerMap2DImpl_h__
#define __L1Trigger_L1THGCal_HGCalTowerMap2DImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"


class HGCalTowerMap2DImpl{

 public:

  HGCalTowerMap2DImpl( const edm::ParameterSet &conf); 
  
  void eventSetup(const edm::EventSetup& es) 
    {
        triggerTools_.setEventSetup(es);
    }

  void buildTowerMapClusters( const std::vector<edm::Ptr<l1t::HGCalTriggerCell>> & triggerCellsPtrs,
			      l1t::HGCalClusterBxCollection & clusters
			      );

  void calibratePt( l1t::HGCalCluster & cluster );  
  
 private:

  static const int kLayersEE_ = 28;
  static const int kLayersFH_ = 12;  
  static const unsigned kLayersBH_ = 12;
  static const unsigned kLayers_ = kLayersEE_+kLayersFH_+kLayersBH_;

  static constexpr double kEtaMin_ = 1.479;
  static constexpr double kEtaMax_ = 3.;
  static constexpr double kPhiMin_ = -M_PI;
  static constexpr double kPhiMax_ = +M_PI;  

  unsigned nEtaBins_;
  unsigned nPhiBins_;
  vector<double> etaBins_;
  vector<double> phiBins_;

  std::vector<double> layerWeights_;

  HGCalTriggerTools triggerTools_;

};



#endif
