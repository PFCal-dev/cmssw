#ifndef HGCalTokens_h
#define HGCalTokens_h

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
//#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
//#include "DataFormats/L1Trigger/interface/EGamma.h"
//#include "DataFormats/L1Trigger/interface/EtSum.h"
//#include "DataFormats/L1Trigger/interface/Jet.h"
//#include "DataFormats/L1Trigger/interface/Tau.h"

#include "CommonTokens.h"

namespace l1t {
   namespace stage2 {
      class HGCalTokens : public CommonTokens {
         public:
            HGCalTokens(const edm::ParameterSet&, edm::ConsumesCollector&);

            inline const edm::EDGetTokenT<HGCalTriggerCellBxCollection>& getHGCalTriggerCellToken() const { return hgcalTrigCellToken_; };

         private:
            edm::EDGetTokenT<HGCalTriggerCellBxCollection> hgcalTrigCellToken_;
      };
   }
}

#endif
