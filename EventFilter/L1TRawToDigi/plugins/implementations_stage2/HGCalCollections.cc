#include "FWCore/Framework/interface/Event.h"

#include "HGCalCollections.h"

namespace l1t {
   namespace stage2 {
      HGCalCollections::~HGCalCollections()
      {
         event_.put(std::move(trigcells_),"HGCalTrigCell");
      }
   }
}
