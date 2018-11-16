#ifndef HGCalCollections_h
#define HGCalCollections_h

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

//#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"
#include "L1TObjectCollections.h"

namespace l1t {
   namespace stage2 {
     class HGCalCollections : public L1TObjectCollections {
         public:
            HGCalCollections(edm::Event& e) :
               L1TObjectCollections(e),
               trigcells_(new HGCalTriggerCellBxCollection()) {};

            ~HGCalCollections() override;

            inline HGCalTriggerCellBxCollection* getHGCTrigCells() { return trigcells_.get(); };

         private:
            std::unique_ptr<HGCalTriggerCellBxCollection> trigcells_;
      };
   }
}

#endif
