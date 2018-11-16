#ifndef L1T_PACKER_STAGE2_HGCALSETUP_H
#define L1T_PACKER_STAGE2_HGCALSETUP_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"

#include "HGCalCollections.h"
#include "HGCalTokens.h"

namespace l1t {
   namespace stage2 {
      class HGCalSetup : public PackingSetup {
         public:
            std::unique_ptr<PackerTokens> registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) override;
            void fillDescription(edm::ParameterSetDescription& desc) override;
            PackerMap getPackers(int fed, unsigned int fw) override;
            void registerProducts(edm::stream::EDProducerBase& prod) override;
            std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override;
            UnpackerMap getUnpackers(int fed, int board, int amc, unsigned int fw) override;
      };
   }
}

#endif
