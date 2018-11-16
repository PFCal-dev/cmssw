#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HGCalTokens.h"

namespace l1t {
   namespace stage2 {
      HGCalTokens::HGCalTokens(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) 
      {
         auto tcelltag = cfg.getParameter<edm::InputTag>("HgcalTriggerCellInputLabel");
         //auto tag = cfg.getParameter<edm::InputTag>("InputLabel");
         hgcalTrigCellToken_ = cc.consumes<HGCalTriggerCellBxCollection>(tcelltag);
         //egammaToken_ = cc.consumes<EGammaBxCollection>(tag);
         //etSumToken_ = cc.consumes<EtSumBxCollection>(tag);
         //jetToken_ = cc.consumes<JetBxCollection>(tag);
         //tauToken_ = cc.consumes<TauBxCollection>(tag);
      }
   }
}
